"""Парсер бинарного формата KN5 (Assetto Corsa).

Формат (по исходникам Actools, AcTools/Kn5File/Kn5Basic.cs):

    "sc6969"                    6 байт magic
    int32 version               (> 5 => дополнительный int32 extra)
    int32 texture_count
        int32 active, string name, uint32 length, length байт данных
    int32 material_count
        string name, string shader, byte blend_mode, bool alpha_tested,
        int32 depth_mode, int32 prop_count,
            (string name, float a, vec2 b, vec3 c, vec4 d) * prop_count
        int32 map_count,
            (string name, int32 slot, string texture) * map_count
    node                          корневой узел, далее рекурсивно

    node:
        int32 class               1 = Base(dummy), 2 = Mesh, 3 = SkinnedMesh
        string name
        int32 children_count
        byte  active
        class 1: float transform[16]
        class 2: byte cast_shadows, byte visible, byte transparent,
                 uint32 vertex_count, vertex[44 байт] * N,
                 uint32 index_count, uint16[N],
                 uint32 material, uint32 layer, float lod_in, float lod_out,
                 vec3 bound_center, float bound_radius, byte is_renderable
        class 3: как Mesh, но vertex 76 байт (веса), у каждого bone
                 (string name, float matrix[16]) и в конце 8 байт.

Строка: int32 длина + байты (может содержать завершающий нуль).

ВАЖНО про матрицы: 16 float в KN5 лежат ровно в том порядке, в котором их
ожидает glTF (column-major массив, трансляция в индексах 12..14). Это не
случайность: KN5 хранит матрицу как A^T в row-major, а это и есть
column-major сериализация A. Поэтому матрица узла переносится в GLB 1:1.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np

KN5_MAGIC = b"sc6969"

NODE_BASE = 1
NODE_MESH = 2
NODE_SKINNED = 3

# 11 float: позиция, нормаль, uv, тангент
VERTEX_DTYPE = np.dtype(
    [
        ("pos", "<f4", (3,)),
        ("nrm", "<f4", (3,)),
        ("uv", "<f4", (2,)),
        ("tan", "<f4", (3,)),
    ]
)
# 19 float: то же + веса скиннинга
SKINNED_DTYPE = np.dtype(
    [
        ("pos", "<f4", (3,)),
        ("nrm", "<f4", (3,)),
        ("uv", "<f4", (2,)),
        ("tan", "<f4", (3,)),
        ("weights", "<f4", (4,)),
        ("bones", "<f4", (4,)),
    ]
)


class Kn5Error(Exception):
    pass


@dataclass
class Texture:
    name: str
    active: bool
    data: bytes

    @property
    def ext(self) -> str:
        head = self.data[:4]
        if head == b"DDS ":
            return "dds"
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if head[:3] == b"\xff\xd8\xff":
            return "jpg"
        return "bin"


@dataclass
class ShaderProperty:
    name: str
    a: float = 0.0
    b: tuple = (0.0, 0.0)
    c: tuple = (0.0, 0.0, 0.0)
    d: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class TextureMapping:
    name: str
    slot: int
    texture: str


@dataclass
class Material:
    name: str
    shader: str
    blend_mode: int = 0
    alpha_tested: bool = False
    depth_mode: int = 0
    properties: dict[str, ShaderProperty] = field(default_factory=dict)
    mappings: list[TextureMapping] = field(default_factory=list)

    def texture(self, name: str) -> str | None:
        for m in self.mappings:
            if m.name == name:
                return m.texture
        return None

    def prop(self, name: str) -> ShaderProperty | None:
        return self.properties.get(name)


@dataclass
class Node:
    name: str
    cls: int
    active: bool = True
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    children: list["Node"] = field(default_factory=list)

    # --- геометрия ---
    vertices: np.ndarray | None = None  # (N, 11) float32
    indices: np.ndarray | None = None  # (M,) uint16
    material: int = -1
    layer: int = 0
    lod_in: float = 0.0
    lod_out: float = 0.0
    bound_center: tuple = (0.0, 0.0, 0.0)
    bound_radius: float = 0.0
    cast_shadows: bool = True
    visible: bool = True
    transparent: bool = False
    renderable: bool = True
    bones: list[tuple[str, np.ndarray]] = field(default_factory=list)
    weights: np.ndarray | None = None

    @property
    def is_mesh(self) -> bool:
        return self.cls in (NODE_MESH, NODE_SKINNED)

    @property
    def transform(self) -> np.ndarray:
        """Аффинная матрица в column-vector конвенции (A = raw^T)."""
        return self.matrix.T

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, name: str) -> "Node | None":
        for n in self.walk():
            if n.name == name:
                return n
        return None

    def find_all(self, *names: str) -> list["Node"]:
        wanted = {n.upper() for n in names}
        return [n for n in self.walk() if n.name.upper() in wanted]


@dataclass
class Kn5:
    version: int = 0
    textures: dict[str, Texture] = field(default_factory=dict)
    materials: list[Material] = field(default_factory=list)
    root: Node | None = None
    path: str = ""

    def material_by_id(self, idx: int) -> Material | None:
        if 0 <= idx < len(self.materials):
            return self.materials[idx]
        return None

    def stats(self) -> dict:
        verts = tris = meshes = 0
        for n in self.root.walk() if self.root else []:
            if n.is_mesh and n.vertices is not None:
                meshes += 1
                verts += len(n.vertices)
                tris += len(n.indices) // 3
        return dict(meshes=meshes, vertices=verts, triangles=tris, materials=len(self.materials), textures=len(self.textures))


class _Reader:
    __slots__ = ("f", "size")

    def __init__(self, f: BinaryIO):
        self.f = f
        f.seek(0, 2)
        self.size = f.tell()
        f.seek(0)

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise Kn5Error(f"Неожиданный конец файла: нужно {n} байт, получено {len(b)}")
        return b

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def byte(self) -> int:
        return self.read(1)[0]

    def boolean(self) -> bool:
        return self.byte() != 0

    def string(self) -> str:
        n = self.i32()
        if n < 0 or n > 1 << 20:
            raise Kn5Error(f"Повреждённая строка KN5 (длина {n})")
        raw = self.read(n)
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def vec(self, n: int) -> tuple:
        return struct.unpack("<" + "f" * n, self.read(4 * n))

    def mat4(self) -> np.ndarray:
        return np.frombuffer(self.read(64), dtype="<f4").astype(np.float64).reshape(4, 4)


def _read_material(r: _Reader) -> Material:
    m = Material(name=r.string(), shader=r.string())
    m.blend_mode = r.byte()
    m.alpha_tested = r.boolean()
    m.depth_mode = r.i32()
    for _ in range(r.i32()):
        p = ShaderProperty(name=r.string(), a=r.f32(), b=r.vec(2), c=r.vec(3), d=r.vec(4))
        m.properties[p.name] = p
    for _ in range(r.i32()):
        m.mappings.append(TextureMapping(name=r.string(), slot=r.i32(), texture=r.string()))
    return m


def _read_node(r: _Reader) -> Node:
    cls = r.i32()
    if cls not in (NODE_BASE, NODE_MESH, NODE_SKINNED):
        raise Kn5Error(f"Неизвестный класс узла KN5: {cls}")
    n = Node(name=r.string(), cls=cls)
    children = r.i32()
    n.active = r.boolean()

    if cls == NODE_BASE:
        n.matrix = r.mat4()
        n.children = [_read_node(r) for _ in range(children)]
        return n

    n.cast_shadows = r.boolean()
    n.visible = r.boolean()
    n.transparent = r.boolean()

    if cls == NODE_SKINNED:
        for _ in range(r.u32()):
            bone_name = r.string()
            n.bones.append((bone_name, r.mat4()))
        vcount = r.u32()
        buf = r.read(76 * vcount)
        arr = np.frombuffer(buf, dtype=SKINNED_DTYPE, count=vcount)
        n.weights = np.stack([arr["weights"], arr["bones"]], axis=1)
        n.vertices = np.concatenate(
            [arr["pos"], arr["nrm"], arr["uv"], arr["tan"]], axis=1
        ).astype(np.float32)
    else:
        vcount = r.u32()
        buf = r.read(44 * vcount)
        arr = np.frombuffer(buf, dtype=VERTEX_DTYPE, count=vcount)
        n.vertices = np.concatenate(
            [arr["pos"], arr["nrm"], arr["uv"], arr["tan"]], axis=1
        ).astype(np.float32)

    icount = r.u32()
    n.indices = np.frombuffer(r.read(2 * icount), dtype="<u2", count=icount)
    n.material = r.u32()
    n.layer = r.u32()

    if cls == NODE_SKINNED:
        # у скин-мешей нет bbox
        n.lod_in = r.f32()
        n.lod_out = r.f32()
        n.renderable = True
        n.bound_center = (0.0, 0.0, 0.0)
        n.bound_radius = 0.0
    else:
        n.lod_in = r.f32()
        n.lod_out = r.f32()
        n.bound_center = r.vec(3)
        n.bound_radius = r.f32()
        n.renderable = r.boolean()

    # у мешей тоже бывают дети (редко, но формат разрешает)
    if children:
        n.children = [_read_node(r) for _ in range(children)]
    return n


def load(path: str) -> Kn5:
    """Прочитать .kn5 в память."""
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.read(6)
        if magic != KN5_MAGIC:
            raise Kn5Error(f"{path}: не KN5 (magic={magic!r})")
        kn5 = Kn5(path=path)
        kn5.version = r.i32()
        if kn5.version > 5:
            r.i32()  # extra

        for _ in range(r.i32()):
            active = r.i32() == 1
            name = r.string()
            length = r.u32()
            data = r.read(length) if length else b""
            if length:
                kn5.textures[name.lower()] = Texture(name=name, active=active, data=data)

        for _ in range(r.i32()):
            m = _read_material(r)
            kn5.materials.append(m)

        kn5.root = _read_node(r)
        return kn5


# --------------------------------------------------------------------------
# Геометрия: мировые трансформации и батчинг
# --------------------------------------------------------------------------


def world_matrix(node_path: list[Node]) -> np.ndarray:
    """A = A0 @ A1 @ ... (column-vector конвенция)."""
    m = np.eye(4)
    for n in node_path:
        m = m @ n.transform
    return m


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def transform_normals(m: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Нормали под произвольной (в т.ч. с масштабом) матрицей: inverse-transpose."""
    n3 = np.asarray(m[:3, :3], dtype=np.float64)
    det = np.linalg.det(n3)
    if abs(det) < 1e-12:
        return np.asarray(nrm, dtype=np.float64)
    it = np.linalg.inv(n3).T
    out = np.asarray(nrm, dtype=np.float64) @ it.T
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return out / ln


def iter_meshes(root: Node, path: tuple[Node, ...] = ()):
    """Обход дерева: (узел, путь_включая_узел, мировая_матрица)."""
    stack = [(root, path + (root,))]
    while stack:
        node, p = stack.pop()
        if node.is_mesh and node.vertices is not None and len(node.indices):
            yield node, p, world_matrix(list(p))
        for c in reversed(node.children):
            stack.append((c, p + (c,)))