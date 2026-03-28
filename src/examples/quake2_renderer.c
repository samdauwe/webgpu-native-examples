/* -------------------------------------------------------------------------- *
 * Quake 2 BSP/PAK Loader
 *
 * Loads Quake 2 PAK archive files and BSP map data. Parses the complete BSP
 * file format including geometry, textures, lightmaps, visibility data, and
 * the BSP tree structure. Prints metadata to verify loading correctness.
 *
 * This is the first phase: data structures and file loading only.
 * Rendering will be added in a subsequent phase.
 *
 * Ref:
 * https://www.flipcode.com/archives/Quake_2_BSP_File_Format.shtml
 * https://github.com/measuredweighed/BSP2OBJ
 * https://github.com/go-gl-legacy/go-quake2
 * -------------------------------------------------------------------------- */

#include "webgpu/wgpu_common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "webgpu/imgui_overlay.h"

#include "core/camera.h"

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <cglm/cglm.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------- *
 * Quake 2 Format Constants
 * -------------------------------------------------------------------------- */

#define Q2_BSP_MAGIC 0x50534249 /* "IBSP" in little-endian */
#define Q2_BSP_VERSION 38
#define Q2_PAK_MAGIC 0x4B434150 /* "PACK" in little-endian */
#define Q2_NUM_LUMPS 19
#define Q2_MAX_MAP_TEXTURES 1024

/* PAK directory entry sizes */
#define Q2_PAK_FILENAME_LEN 56
#define Q2_PAK_ENTRY_SIZE 64

/* WAL texture header size */
#define Q2_WAL_NAME_LEN 32

/* Lightmap constants */
#define Q2_LIGHTMAP_BLOCK_SIZE 16 /* Luxel granularity in world units */
#define Q2_LIGHTMAP_ATLAS_SIZE 512

/* Surface flags */
#define Q2_SURF_LIGHT 0x0001
#define Q2_SURF_SLICK 0x0002
#define Q2_SURF_SKY 0x0004
#define Q2_SURF_WARP 0x0008
#define Q2_SURF_TRANS33 0x0010
#define Q2_SURF_TRANS66 0x0020
#define Q2_SURF_FLOWING 0x0040
#define Q2_SURF_NODRAW 0x0080

/* -------------------------------------------------------------------------- *
 * BSP Lump Indices
 * -------------------------------------------------------------------------- */

typedef enum {
  Q2_LUMP_ENTITIES     = 0,
  Q2_LUMP_PLANES       = 1,
  Q2_LUMP_VERTICES     = 2,
  Q2_LUMP_VISIBILITY   = 3,
  Q2_LUMP_NODES        = 4,
  Q2_LUMP_TEXINFO      = 5,
  Q2_LUMP_FACES        = 6,
  Q2_LUMP_LIGHTMAPS    = 7,
  Q2_LUMP_LEAVES       = 8,
  Q2_LUMP_LEAF_FACES   = 9,
  Q2_LUMP_LEAF_BRUSHES = 10,
  Q2_LUMP_EDGES        = 11,
  Q2_LUMP_FACE_EDGES   = 12,
  Q2_LUMP_MODELS       = 13,
  Q2_LUMP_BRUSHES      = 14,
  Q2_LUMP_BRUSH_SIDES  = 15,
  Q2_LUMP_POP          = 16,
  Q2_LUMP_AREAS        = 17,
  Q2_LUMP_AREA_PORTALS = 18,
} q2_lump_index_t;

/* -------------------------------------------------------------------------- *
 * BSP Data Structures (all packed, little-endian)
 *
 * Field sizes and offsets match the Quake 2 BSP specification exactly.
 * -------------------------------------------------------------------------- */

/* Lump directory entry (8 bytes) */
typedef struct {
  uint32_t offset; /* Byte offset from file start */
  uint32_t length; /* Length in bytes */
} q2_lump_t;

/* BSP file header (160 bytes) */
typedef struct {
  uint32_t magic;                /* "IBSP" = 0x50534249 */
  uint32_t version;              /* 38 */
  q2_lump_t lumps[Q2_NUM_LUMPS]; /* 19 lump directory entries */
} q2_bsp_header_t;

/* Vertex / 3D point (12 bytes) */
typedef struct {
  float x, y, z;
} q2_vertex_t;

/* Edge - pair of vertex indices (4 bytes) */
typedef struct {
  uint16_t v1, v2;
} q2_edge_t;

/* Plane equation (20 bytes) */
typedef struct {
  float normal[3]; /* Normal vector (Nx, Ny, Nz) */
  float dist;      /* Distance from origin */
  uint32_t type;   /* Axis type: 0=X, 1=Y, 2=Z, 3+=arbitrary */
} q2_plane_t;

/* Face / surface (20 bytes) */
typedef struct {
  uint16_t plane;             /* Plane index */
  uint16_t plane_side;        /* 0 = front, 1 = back  */
  uint32_t first_edge;        /* Index into face-edge table */
  uint16_t num_edges;         /* Number of edges */
  uint16_t texinfo;           /* Texture info index */
  uint8_t lightmap_styles[4]; /* Lightmap style indices (0xFF = unused) */
  uint32_t lightmap_offset;   /* Byte offset into lightmap lump */
} q2_face_t;

/* Texture info (76 bytes) */
typedef struct {
  float u_axis[3];                    /* Texture U basis vector */
  float u_offset;                     /* U offset */
  float v_axis[3];                    /* Texture V basis vector */
  float v_offset;                     /* V offset */
  uint32_t flags;                     /* Surface flags (SKY, WARP, etc.) */
  uint32_t value;                     /* Light intensity / animation speed */
  char texture_name[Q2_WAL_NAME_LEN]; /* Null-terminated texture path */
  int32_t next_texinfo;               /* Animation chain, -1 = end */
} q2_texinfo_t;

/* BSP tree internal node (28 bytes) */
typedef struct {
  uint32_t plane;      /* Splitting plane index */
  int32_t front_child; /* Front child (negative = -(leaf+1)) */
  int32_t back_child;  /* Back child (negative = -(leaf+1)) */
  int16_t bbox_min[3]; /* Bounding box minimum */
  int16_t bbox_max[3]; /* Bounding box maximum */
  uint16_t first_face; /* First face in this node */
  uint16_t num_faces;  /* Number of faces */
} q2_node_t;

/* BSP tree leaf node (28 bytes) */
typedef struct {
  uint32_t brush_or;         /* Brush OR contents */
  uint16_t cluster;          /* Visibility cluster (0xFFFF = no vis) */
  uint16_t area;             /* Area portal index */
  int16_t bbox_min[3];       /* Bounding box minimum */
  int16_t bbox_max[3];       /* Bounding box maximum */
  uint16_t first_leaf_face;  /* Index into leaf-face table */
  uint16_t num_leaf_faces;   /* Number of leaf faces */
  uint16_t first_leaf_brush; /* Index into leaf-brush table */
  uint16_t num_leaf_brushes; /* Number of leaf brushes */
} q2_leaf_t;

/* Visibility offset entry (8 bytes) */
typedef struct {
  uint32_t pvs; /* Offset to PVS data */
  uint32_t phs; /* Offset to PHS data */
} q2_vis_offset_t;

/* BSP sub-model (48 bytes) */
typedef struct {
  float mins[3];      /* Bounding box minimum */
  float maxs[3];      /* Bounding box maximum */
  float origin[3];    /* Model origin */
  int32_t head_node;  /* Index of first BSP node */
  int32_t first_face; /* First face index */
  int32_t num_faces;  /* Number of faces */
} q2_model_t;

/* Brush (12 bytes) */
typedef struct {
  int32_t first_side; /* First brush side index */
  int32_t num_sides;  /* Number of sides */
  int32_t contents;   /* Content flags */
} q2_brush_t;

/* Brush side (4 bytes) */
typedef struct {
  uint16_t plane;  /* Plane index */
  int16_t texinfo; /* Texture info index */
} q2_brush_side_t;

/* Area (8 bytes) */
typedef struct {
  int32_t num_area_portals;  /* Number of portals in this area */
  int32_t first_area_portal; /* Index of first area portal */
} q2_area_t;

/* Area portal (8 bytes) */
typedef struct {
  int32_t portal_num; /* Portal number */
  int32_t other_area; /* Area on the other side */
} q2_area_portal_t;

/* WAL texture header (100 bytes) */
typedef struct {
  char name[Q2_WAL_NAME_LEN];      /* Texture name */
  uint32_t width;                  /* Width in pixels */
  uint32_t height;                 /* Height in pixels */
  int32_t offsets[4];              /* Offsets to 4 mip-levels */
  char next_name[Q2_WAL_NAME_LEN]; /* Next texture in anim chain */
  uint32_t flags;                  /* Surface flags */
  uint32_t contents;               /* Content flags */
  uint32_t value;                  /* Light value */
} q2_wal_header_t;

/* -------------------------------------------------------------------------- *
 * PAK Archive Structures
 * -------------------------------------------------------------------------- */

/* PAK file header (12 bytes) */
typedef struct {
  uint32_t magic;  /* "PACK" = 0x4B434150 */
  uint32_t offset; /* Byte offset to directory */
  uint32_t length; /* Total directory size in bytes */
} q2_pak_header_t;

/* PAK directory entry (64 bytes) */
typedef struct {
  char filename[Q2_PAK_FILENAME_LEN]; /* Null-terminated file path */
  uint32_t offset;                    /* File data offset in PAK */
  uint32_t length;                    /* File size in bytes */
} q2_pak_entry_t;

/* -------------------------------------------------------------------------- *
 * In-Memory Parsed Data
 * -------------------------------------------------------------------------- */

/* Complete parsed BSP map */
typedef struct {
  /* Header */
  q2_bsp_header_t header;

  /* Lump data arrays */
  char* entities; /* Entity string (null-terminated) */
  q2_plane_t* planes;
  q2_vertex_t* vertices;
  q2_node_t* nodes;
  q2_texinfo_t* texinfos;
  q2_face_t* faces;
  uint8_t* lightmap_data; /* Raw RGB lightmap pixels */
  q2_leaf_t* leaves;
  uint16_t* leaf_faces;   /* Leaf-face index table */
  uint16_t* leaf_brushes; /* Leaf-brush index table */
  q2_edge_t* edges;
  int32_t* face_edges; /* Signed face-edge indices */
  q2_model_t* models;
  q2_brush_t* brushes;
  q2_brush_side_t* brush_sides;
  q2_area_t* areas;
  q2_area_portal_t* area_portals;

  /* Visibility data */
  uint32_t num_vis_clusters;
  q2_vis_offset_t* vis_offsets; /* Per-cluster PVS/PHS offsets */
  uint8_t* vis_data;            /* Raw compressed visibility data */
  uint32_t vis_data_size;

  /* Element counts (derived from lump sizes) */
  uint32_t num_entities_len;
  uint32_t num_planes;
  uint32_t num_vertices;
  uint32_t num_nodes;
  uint32_t num_texinfos;
  uint32_t num_faces;
  uint32_t num_lightmap_bytes;
  uint32_t num_leaves;
  uint32_t num_leaf_faces;
  uint32_t num_leaf_brushes;
  uint32_t num_edges;
  uint32_t num_face_edges;
  uint32_t num_models;
  uint32_t num_brushes;
  uint32_t num_brush_sides;
  uint32_t num_areas;
  uint32_t num_area_portals;
} q2_bsp_map_t;

/* PAK archive with directory lookup */
typedef struct {
  uint8_t* file_data;      /* Entire PAK file in memory */
  uint32_t file_size;      /* Total PAK file size */
  q2_pak_entry_t* entries; /* Directory entries */
  uint32_t num_entries;    /* Number of files in PAK */
} q2_pak_t;

/* -------------------------------------------------------------------------- *
 * Quake 2 Color Palette (256 RGB entries)
 *
 * Standard palette used by WAL textures. Each byte in a WAL texture indexes
 * into this palette to produce an RGB pixel.
 * -------------------------------------------------------------------------- */

// clang-format off
static const uint8_t q2_palette[256][3] = {
  {  0,  0,  0}, { 15, 15, 15}, { 31, 31, 31}, { 47, 47, 47},
  { 63, 63, 63}, { 75, 75, 75}, { 91, 91, 91}, {107,107,107},
  {123,123,123}, {139,139,139}, {155,155,155}, {171,171,171},
  {187,187,187}, {203,203,203}, {219,219,219}, {235,235,235},
  { 99, 75, 35}, { 91, 67, 31}, { 83, 63, 31}, { 79, 59, 27},
  { 71, 55, 27}, { 63, 47, 23}, { 59, 43, 23}, { 51, 39, 19},
  { 47, 35, 19}, { 43, 31, 19}, { 39, 27, 15}, { 35, 23, 15},
  { 27, 19, 11}, { 23, 15, 11}, { 19, 15,  7}, { 15, 11,  7},
  { 95, 95,111}, { 91, 91,103}, { 91, 83, 95}, { 87, 79, 91},
  { 83, 75, 83}, { 79, 71, 75}, { 71, 63, 67}, { 63, 59, 59},
  { 59, 55, 55}, { 51, 47, 47}, { 47, 43, 43}, { 39, 39, 39},
  { 35, 35, 35}, { 27, 27, 27}, { 23, 23, 23}, { 19, 19, 19},
  {143,119, 83}, {123, 99, 67}, {115, 91, 59}, {103, 79, 47},
  {207,151, 75}, {167,123, 59}, {139,103, 47}, {111, 83, 39},
  {235,159, 39}, {203,139, 35}, {175,119, 31}, {147, 99, 27},
  {119, 79, 23}, { 91, 59, 15}, { 63, 39, 11}, { 35, 23,  7},
  {167, 59, 43}, {159, 47, 35}, {151, 43, 27}, {139, 39, 19},
  {127, 31, 15}, {115, 23, 11}, {103, 23,  7}, { 87, 19,  0},
  { 75, 15,  0}, { 67, 15,  0}, { 59, 15,  0}, { 51, 11,  0},
  { 43, 11,  0}, { 35, 11,  0}, { 27,  7,  0}, { 19,  7,  0},
  {123, 95, 75}, {115, 87, 67}, {107, 83, 63}, {103, 79, 59},
  { 95, 71, 55}, { 87, 67, 51}, { 83, 63, 47}, { 75, 55, 43},
  { 67, 51, 39}, { 63, 47, 35}, { 55, 39, 27}, { 47, 35, 23},
  { 39, 27, 19}, { 31, 23, 15}, { 23, 15, 11}, { 15, 11,  7},
  {111, 59, 23}, { 95, 55, 23}, { 83, 47, 23}, { 67, 43, 23},
  { 55, 35, 19}, { 39, 27, 15}, { 27, 19, 11}, { 15, 11,  7},
  {179, 91, 79}, {191,123,111}, {203,155,147}, {215,187,183},
  {203,215,223}, {179,199,211}, {159,183,195}, {135,167,183},
  {115,151,167}, { 91,135,155}, { 71,119,139}, { 47,103,127},
  { 23, 83,111}, { 19, 75,103}, { 15, 67, 91}, { 11, 63, 83},
  {  7, 55, 75}, {  7, 47, 63}, {  7, 39, 51}, {  0, 31, 43},
  {  0, 23, 31}, {  0, 15, 19}, {  0,  7, 11}, {  0,  0,  0},
  {139, 87, 87}, {131, 79, 79}, {123, 71, 71}, {115, 67, 67},
  {107, 59, 59}, { 99, 51, 51}, { 91, 47, 47}, { 87, 43, 43},
  { 75, 35, 35}, { 63, 31, 31}, { 51, 27, 27}, { 43, 19, 19},
  { 31, 15, 15}, { 19, 11, 11}, { 11,  7,  7}, {  0,  0,  0},
  {151,159,123}, {143,151,115}, {135,139,107}, {127,131, 99},
  {119,123, 95}, {115,115, 87}, {107,107, 79}, { 99, 99, 71},
  { 91, 91, 67}, { 79, 79, 59}, { 67, 67, 51}, { 55, 55, 43},
  { 47, 47, 35}, { 35, 35, 27}, { 23, 23, 19}, { 15, 15, 11},
  {159, 75, 63}, {147, 67, 55}, {139, 59, 47}, {127, 55, 39},
  {119, 47, 35}, {107, 43, 27}, { 99, 35, 23}, { 87, 31, 19},
  { 79, 27, 15}, { 67, 23, 11}, { 55, 19, 11}, { 43, 15,  7},
  { 31, 11,  7}, { 23,  7,  0}, { 11,  0,  0}, {  0,  0,  0},
  {119,123,207}, {111,115,195}, {103,107,183}, { 99, 99,167},
  { 91, 91,155}, { 83, 87,143}, { 75, 79,127}, { 71, 71,115},
  { 63, 63,103}, { 55, 55, 87}, { 47, 47, 75}, { 39, 39, 63},
  { 35, 31, 47}, { 27, 23, 35}, { 19, 15, 23}, { 11,  7,  7},
  {155,171,123}, {143,159,111}, {135,151, 99}, {123,139, 87},
  {115,131, 75}, {103,119, 67}, { 95,111, 59}, { 87,103, 51},
  { 75, 91, 39}, { 63, 79, 27}, { 55, 67, 19}, { 47, 59, 11},
  { 35, 47,  7}, { 27, 35,  0}, { 19, 23,  0}, { 11, 15,  0},
  {  0,255,  0}, { 35,231, 15}, { 63,211, 27}, { 83,187, 39},
  { 95,167, 47}, { 95,143, 51}, { 95,123, 51}, {255,255,255},
  {255,255,211}, {255,255,167}, {255,255,127}, {255,255, 83},
  {255,255, 39}, {255,235, 31}, {255,215, 23}, {255,191, 15},
  {255,171,  7}, {255,147,  0}, {239,127,  0}, {227,107,  0},
  {211, 87,  0}, {199, 71,  0}, {183, 59,  0}, {171, 43,  0},
  {155, 31,  0}, {143, 23,  0}, {127, 15,  0}, {115,  7,  0},
  { 95,  0,  0}, { 71,  0,  0}, { 47,  0,  0}, { 27,  0,  0},
  {239,  0,  0}, { 55, 55,255}, {255,  0,  0}, {  0,  0,255},
  { 43, 43, 35}, { 27, 27, 23}, { 19, 19, 15}, {235,151,127},
  {195,115, 83}, {159, 87, 51}, {123, 63, 27}, {235,211,199},
  {199,171,155}, {167,139,119}, {135,107, 87}, {159, 91, 83},
};
// clang-format on

/* -------------------------------------------------------------------------- *
 * File I/O Helpers
 * -------------------------------------------------------------------------- */

/**
 * @brief Query file size using stat(). Returns 0 on failure.
 */
static size_t file_get_size(const char* path)
{
  struct stat st;
  if (stat(path, &st) != 0) {
    fprintf(stderr, "[ERROR] Cannot stat file: %s\n", path);
    return 0;
  }
  return (size_t)st.st_size;
}

/**
 * @brief Load an entire file into a dynamically allocated buffer.
 *        Caller must free() the returned pointer.
 *
 * @param path     File path.
 * @param out_size Receives the file size in bytes.
 * @return Allocated buffer with file contents, or NULL on failure.
 */
static uint8_t* file_load(const char* path, size_t* out_size)
{
  *out_size = 0;

  size_t size = file_get_size(path);
  if (size == 0) {
    return NULL;
  }

  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[ERROR] Cannot open file: %s\n", path);
    return NULL;
  }

  uint8_t* buffer = (uint8_t*)malloc(size);
  if (!buffer) {
    fprintf(stderr, "[ERROR] Failed to allocate %zu bytes for: %s\n", size,
            path);
    fclose(f);
    return NULL;
  }

  size_t read = fread(buffer, 1, size, f);
  fclose(f);

  if (read != size) {
    fprintf(stderr, "[ERROR] Read %zu of %zu bytes from: %s\n", read, size,
            path);
    free(buffer);
    return NULL;
  }

  *out_size = size;
  return buffer;
}

/* -------------------------------------------------------------------------- *
 * PAK Archive Loading
 * -------------------------------------------------------------------------- */

/**
 * @brief Load and parse a Quake 2 PAK archive.
 *
 * The entire file is loaded into memory. The directory is parsed in-place
 * without additional allocations for entry data.
 *
 * @param path  Path to the .pak file.
 * @param pak   Output PAK structure.
 * @return true on success.
 */
static bool q2_pak_load(const char* path, q2_pak_t* pak)
{
  memset(pak, 0, sizeof(*pak));

  pak->file_data = file_load(path, (size_t*)&pak->file_size);
  if (!pak->file_data) {
    return false;
  }

  if (pak->file_size < sizeof(q2_pak_header_t)) {
    fprintf(stderr, "[ERROR] PAK file too small: %u bytes\n", pak->file_size);
    free(pak->file_data);
    pak->file_data = NULL;
    return false;
  }

  /* Parse header */
  const q2_pak_header_t* header = (const q2_pak_header_t*)pak->file_data;

  if (header->magic != Q2_PAK_MAGIC) {
    fprintf(stderr, "[ERROR] Invalid PAK magic: 0x%08X (expected 0x%08X)\n",
            header->magic, Q2_PAK_MAGIC);
    free(pak->file_data);
    pak->file_data = NULL;
    return false;
  }

  /* Validate directory bounds */
  if (header->offset + header->length > pak->file_size) {
    fprintf(stderr, "[ERROR] PAK directory extends beyond file\n");
    free(pak->file_data);
    pak->file_data = NULL;
    return false;
  }

  /* Point entries directly into the loaded file buffer (zero-copy) */
  pak->entries     = (q2_pak_entry_t*)(pak->file_data + header->offset);
  pak->num_entries = header->length / Q2_PAK_ENTRY_SIZE;

  printf("[PAK] Loaded: %s\n", path);
  printf("[PAK]   File size:    %u bytes (%.2f MB)\n", pak->file_size,
         pak->file_size / (1024.0 * 1024.0));
  printf("[PAK]   Dir offset:   %u\n", header->offset);
  printf("[PAK]   Dir size:     %u bytes\n", header->length);
  printf("[PAK]   File count:   %u\n", pak->num_entries);

  return true;
}

/**
 * @brief Find a file entry in the PAK archive by path.
 *
 * @param pak      Loaded PAK archive.
 * @param filename File path to search for (e.g. "maps/demo1.bsp").
 * @return Pointer to the directory entry, or NULL if not found.
 */
static const q2_pak_entry_t* q2_pak_find(const q2_pak_t* pak,
                                         const char* filename)
{
  for (uint32_t i = 0; i < pak->num_entries; i++) {
    if (strncmp(pak->entries[i].filename, filename, Q2_PAK_FILENAME_LEN) == 0) {
      return &pak->entries[i];
    }
  }
  return NULL;
}

/**
 * @brief Get a pointer to file data within the PAK buffer.
 *
 * @param pak   Loaded PAK archive.
 * @param entry Directory entry.
 * @param size  Receives the file size.
 * @return Pointer into the PAK buffer, or NULL on bounds error.
 */
static const uint8_t* q2_pak_get_data(const q2_pak_t* pak,
                                      const q2_pak_entry_t* entry,
                                      uint32_t* size)
{
  if (entry->offset + entry->length > pak->file_size) {
    fprintf(stderr, "[ERROR] PAK entry '%s' extends beyond file\n",
            entry->filename);
    *size = 0;
    return NULL;
  }
  *size = entry->length;
  return pak->file_data + entry->offset;
}

/**
 * @brief Free PAK archive resources.
 */
static void q2_pak_destroy(q2_pak_t* pak)
{
  if (pak->file_data) {
    free(pak->file_data);
  }
  memset(pak, 0, sizeof(*pak));
}

/* -------------------------------------------------------------------------- *
 * BSP Loading
 * -------------------------------------------------------------------------- */

/**
 * @brief Helper macro to parse a typed array from a BSP lump.
 *
 * Points directly into the source buffer (zero-copy). Calculates element
 * count from (lump_length / element_size).
 */
#define BSP_PARSE_LUMP(map, data, lump_idx, field, count_field, type)          \
  do {                                                                         \
    const q2_lump_t* _l = &(map)->header.lumps[lump_idx];                      \
    (map)->count_field  = _l->length / (uint32_t)sizeof(type);                 \
    (map)->field        = (type*)((data) + _l->offset);                        \
  } while (0)

/**
 * @brief Parse a BSP file from a memory buffer (zero-copy).
 *
 * The buffer must remain valid for the lifetime of the map structure since
 * all lump pointers reference directly into the buffer.
 *
 * @param data      Pointer to BSP file data in memory.
 * @param data_size Size of the buffer in bytes.
 * @param map       Output BSP map structure.
 * @return true on success.
 */
static bool q2_bsp_parse(const uint8_t* data, uint32_t data_size,
                         q2_bsp_map_t* map)
{
  memset(map, 0, sizeof(*map));

  if (data_size < sizeof(q2_bsp_header_t)) {
    fprintf(stderr, "[ERROR] BSP data too small: %u bytes\n", data_size);
    return false;
  }

  /* Copy header (we need our own copy since we'll modify pointers) */
  memcpy(&map->header, data, sizeof(q2_bsp_header_t));

  /* Validate magic and version */
  if (map->header.magic != Q2_BSP_MAGIC) {
    fprintf(stderr, "[ERROR] Invalid BSP magic: 0x%08X (expected 0x%08X)\n",
            map->header.magic, Q2_BSP_MAGIC);
    return false;
  }
  if (map->header.version != Q2_BSP_VERSION) {
    fprintf(stderr, "[ERROR] Invalid BSP version: %u (expected %u)\n",
            map->header.version, Q2_BSP_VERSION);
    return false;
  }

  /* Validate all lumps fit within the data buffer */
  for (int i = 0; i < Q2_NUM_LUMPS; i++) {
    const q2_lump_t* l = &map->header.lumps[i];
    if (l->offset + l->length > data_size) {
      fprintf(stderr, "[ERROR] Lump %d extends beyond data (%u + %u > %u)\n", i,
              l->offset, l->length, data_size);
      return false;
    }
  }

  /* Entities (raw text, needs null termination) */
  {
    const q2_lump_t* l    = &map->header.lumps[Q2_LUMP_ENTITIES];
    map->num_entities_len = l->length;
    if (l->length > 0) {
      map->entities = (char*)malloc(l->length + 1);
      if (map->entities) {
        memcpy(map->entities, data + l->offset, l->length);
        map->entities[l->length] = '\0';
      }
    }
  }

  /* Parse typed lump arrays (zero-copy into data buffer) */
  BSP_PARSE_LUMP(map, data, Q2_LUMP_PLANES, planes, num_planes, q2_plane_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_VERTICES, vertices, num_vertices,
                 q2_vertex_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_NODES, nodes, num_nodes, q2_node_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_TEXINFO, texinfos, num_texinfos,
                 q2_texinfo_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_FACES, faces, num_faces, q2_face_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_LEAVES, leaves, num_leaves, q2_leaf_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_LEAF_FACES, leaf_faces, num_leaf_faces,
                 uint16_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_LEAF_BRUSHES, leaf_brushes,
                 num_leaf_brushes, uint16_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_EDGES, edges, num_edges, q2_edge_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_FACE_EDGES, face_edges, num_face_edges,
                 int32_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_MODELS, models, num_models, q2_model_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_BRUSHES, brushes, num_brushes, q2_brush_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_BRUSH_SIDES, brush_sides, num_brush_sides,
                 q2_brush_side_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_AREAS, areas, num_areas, q2_area_t);
  BSP_PARSE_LUMP(map, data, Q2_LUMP_AREA_PORTALS, area_portals,
                 num_area_portals, q2_area_portal_t);

  /* Lightmap data (raw bytes) */
  {
    const q2_lump_t* l      = &map->header.lumps[Q2_LUMP_LIGHTMAPS];
    map->num_lightmap_bytes = l->length;
    map->lightmap_data      = (uint8_t*)(data + l->offset);
  }

  /* Visibility data */
  {
    const q2_lump_t* l = &map->header.lumps[Q2_LUMP_VISIBILITY];
    if (l->length >= sizeof(uint32_t)) {
      /* First 4 bytes = number of clusters */
      map->num_vis_clusters = *(const uint32_t*)(data + l->offset);

      /* Followed by vis_offset entries and RLE data */
      uint32_t offsets_size
        = map->num_vis_clusters * (uint32_t)sizeof(q2_vis_offset_t);
      if (sizeof(uint32_t) + offsets_size <= l->length) {
        map->vis_offsets
          = (q2_vis_offset_t*)(data + l->offset + sizeof(uint32_t));
        map->vis_data
          = (uint8_t*)(data + l->offset + sizeof(uint32_t) + offsets_size);
        map->vis_data_size = l->length - sizeof(uint32_t) - offsets_size;
      }
    }
  }

  return true;
}

/**
 * @brief Free BSP map resources. Only frees the entity string
 *        (all other pointers are into the source data buffer).
 */
static void q2_bsp_destroy(q2_bsp_map_t* map)
{
  if (map->entities) {
    free(map->entities);
  }
  memset(map, 0, sizeof(*map));
}

/* -------------------------------------------------------------------------- *
 * BSP Metadata Printing
 * -------------------------------------------------------------------------- */

static const char* q2_lump_name(int index)
{
  static const char* names[Q2_NUM_LUMPS] = {
    "Entities",     "Planes", "Vertices",   "Visibility",   "Nodes",
    "Texture Info", "Faces",  "Lightmaps",  "Leaves",       "Leaf Faces",
    "Leaf Brushes", "Edges",  "Face Edges", "Models",       "Brushes",
    "Brush Sides",  "Pop",    "Areas",      "Area Portals",
  };
  if (index >= 0 && index < Q2_NUM_LUMPS) {
    return names[index];
  }
  return "Unknown";
}

static void q2_bsp_print_metadata(const q2_bsp_map_t* map)
{
  printf("\n");
  printf("=== BSP File Metadata ===\n");
  printf("  Magic:        IBSP (0x%08X)\n", map->header.magic);
  printf("  Version:      %u\n", map->header.version);
  printf("\n");

  printf("--- Lump Directory ---\n");
  printf("  %-16s  %10s  %10s  %10s\n", "Lump", "Offset", "Length", "Elements");
  printf("  %-16s  %10s  %10s  %10s\n", "----", "------", "------", "--------");

  /* Element sizes for each lump type */
  static const uint32_t elem_sizes[Q2_NUM_LUMPS] = {
    1,                        /* Entities (bytes) */
    sizeof(q2_plane_t),       /* Planes */
    sizeof(q2_vertex_t),      /* Vertices */
    1,                        /* Visibility (raw bytes) */
    sizeof(q2_node_t),        /* Nodes */
    sizeof(q2_texinfo_t),     /* TexInfo */
    sizeof(q2_face_t),        /* Faces */
    1,                        /* Lightmaps (raw RGB bytes) */
    sizeof(q2_leaf_t),        /* Leaves */
    sizeof(uint16_t),         /* Leaf Faces */
    sizeof(uint16_t),         /* Leaf Brushes */
    sizeof(q2_edge_t),        /* Edges */
    sizeof(int32_t),          /* Face Edges */
    sizeof(q2_model_t),       /* Models */
    sizeof(q2_brush_t),       /* Brushes */
    sizeof(q2_brush_side_t),  /* Brush Sides */
    1,                        /* Pop */
    sizeof(q2_area_t),        /* Areas */
    sizeof(q2_area_portal_t), /* Area Portals */
  };

  for (int i = 0; i < Q2_NUM_LUMPS; i++) {
    const q2_lump_t* l = &map->header.lumps[i];
    uint32_t elems
      = (elem_sizes[i] > 1) ? (l->length / elem_sizes[i]) : l->length;
    printf("  %-16s  %10u  %10u  %10u\n", q2_lump_name(i), l->offset, l->length,
           elems);
  }

  printf("\n--- Parsed Counts ---\n");
  printf("  Planes:       %u\n", map->num_planes);
  printf("  Vertices:     %u\n", map->num_vertices);
  printf("  Nodes:        %u\n", map->num_nodes);
  printf("  TexInfos:     %u\n", map->num_texinfos);
  printf("  Faces:        %u\n", map->num_faces);
  printf("  Lightmap:     %u bytes (%.1f KB)\n", map->num_lightmap_bytes,
         map->num_lightmap_bytes / 1024.0);
  printf("  Leaves:       %u\n", map->num_leaves);
  printf("  Leaf Faces:   %u\n", map->num_leaf_faces);
  printf("  Leaf Brushes: %u\n", map->num_leaf_brushes);
  printf("  Edges:        %u\n", map->num_edges);
  printf("  Face Edges:   %u\n", map->num_face_edges);
  printf("  Models:       %u\n", map->num_models);
  printf("  Brushes:      %u\n", map->num_brushes);
  printf("  Brush Sides:  %u\n", map->num_brush_sides);
  printf("  Areas:        %u\n", map->num_areas);
  printf("  Area Portals: %u\n", map->num_area_portals);

  /* Visibility info */
  if (map->num_vis_clusters > 0) {
    printf("\n--- Visibility ---\n");
    printf("  Clusters:     %u\n", map->num_vis_clusters);
    printf("  Vis data:     %u bytes\n", map->vis_data_size);
  }

  /* Model 0 = world model bounding box */
  if (map->num_models > 0) {
    const q2_model_t* world = &map->models[0];
    printf("\n--- World Model (model 0) ---\n");
    printf("  Min: (%.1f, %.1f, %.1f)\n", world->mins[0], world->mins[1],
           world->mins[2]);
    printf("  Max: (%.1f, %.1f, %.1f)\n", world->maxs[0], world->maxs[1],
           world->maxs[2]);
    printf("  Origin: (%.1f, %.1f, %.1f)\n", world->origin[0], world->origin[1],
           world->origin[2]);
    printf("  Faces: %d (starting at %d)\n", world->num_faces,
           world->first_face);
  }

  /* Print unique texture names */
  if (map->num_texinfos > 0) {
    printf("\n--- Unique Textures ---\n");

    /* Simple dedup using a small visited array */
    char seen[Q2_MAX_MAP_TEXTURES][Q2_WAL_NAME_LEN];
    uint32_t seen_count = 0;
    uint32_t sky_count  = 0;

    for (uint32_t i = 0; i < map->num_texinfos && i < Q2_MAX_MAP_TEXTURES;
         i++) {
      const char* name = map->texinfos[i].texture_name;
      if (name[0] == '\0') {
        continue;
      }

      /* Check if already seen */
      bool found = false;
      for (uint32_t j = 0; j < seen_count; j++) {
        if (strncmp(seen[j], name, Q2_WAL_NAME_LEN) == 0) {
          found = true;
          break;
        }
      }
      if (!found && seen_count < Q2_MAX_MAP_TEXTURES) {
        strncpy(seen[seen_count], name, Q2_WAL_NAME_LEN - 1);
        seen[seen_count][Q2_WAL_NAME_LEN - 1] = '\0';
        seen_count++;

        if (map->texinfos[i].flags & Q2_SURF_SKY) {
          sky_count++;
        }
      }
    }

    printf("  Total unique: %u", seen_count);
    if (sky_count > 0) {
      printf(" (%u sky)", sky_count);
    }
    printf("\n");

    /* Print first 20 texture names as sample */
    uint32_t print_max = seen_count < 20 ? seen_count : 20;
    for (uint32_t i = 0; i < print_max; i++) {
      printf("    [%3u] %s\n", i, seen[i]);
    }
    if (seen_count > print_max) {
      printf("    ... and %u more\n", seen_count - print_max);
    }
  }

  printf("\n");
}

/* -------------------------------------------------------------------------- *
 * PAK Directory Printing
 * -------------------------------------------------------------------------- */

static void q2_pak_print_summary(const q2_pak_t* pak)
{
  printf("\n=== PAK Archive Summary ===\n");

  /* Count file types */
  uint32_t bsp_count = 0, wal_count = 0, pcx_count = 0;
  uint32_t mdl_count = 0, wav_count = 0, other_count = 0;
  uint64_t total_size = 0;

  for (uint32_t i = 0; i < pak->num_entries; i++) {
    const char* name = pak->entries[i].filename;
    uint32_t len     = (uint32_t)strnlen(name, Q2_PAK_FILENAME_LEN);
    total_size += pak->entries[i].length;

    if (len > 4) {
      const char* ext = name + len - 4;
      if (strncmp(ext, ".bsp", 4) == 0) {
        bsp_count++;
      }
      else if (strncmp(ext, ".wal", 4) == 0) {
        wal_count++;
      }
      else if (strncmp(ext, ".pcx", 4) == 0) {
        pcx_count++;
      }
      else if (strncmp(ext, ".md2", 4) == 0 || strncmp(ext, ".sp2", 4) == 0) {
        mdl_count++;
      }
      else if (strncmp(ext, ".wav", 4) == 0) {
        wav_count++;
      }
      else {
        other_count++;
      }
    }
    else {
      other_count++;
    }
  }

  printf("  Total files:      %u\n", pak->num_entries);
  printf("  Total data size:  %.2f MB\n", total_size / (1024.0 * 1024.0));
  printf("  BSP maps:         %u\n", bsp_count);
  printf("  WAL textures:     %u\n", wal_count);
  printf("  PCX images:       %u\n", pcx_count);
  printf("  Models (MD2/SP2): %u\n", mdl_count);
  printf("  Sound files:      %u\n", wav_count);
  printf("  Other:            %u\n", other_count);

  /* List all BSP maps found */
  if (bsp_count > 0) {
    printf("\n--- BSP Maps ---\n");
    for (uint32_t i = 0; i < pak->num_entries; i++) {
      const char* name = pak->entries[i].filename;
      uint32_t len     = (uint32_t)strnlen(name, Q2_PAK_FILENAME_LEN);
      if (len > 4 && strncmp(name + len - 4, ".bsp", 4) == 0) {
        printf("  [%4u] %-40s %8u bytes (%.2f MB)\n", i, name,
               pak->entries[i].length,
               pak->entries[i].length / (1024.0 * 1024.0));
      }
    }
  }

  printf("\n");
}

/* -------------------------------------------------------------------------- *
 * WAL Texture Info Printing (from PAK)
 * -------------------------------------------------------------------------- */

static void q2_wal_print_info(const uint8_t* data, uint32_t size,
                              const char* name)
{
  (void)q2_palette; /* Used in rendering phase */

  if (size < sizeof(q2_wal_header_t)) {
    printf("  [WAL] %s: too small (%u bytes)\n", name, size);
    return;
  }

  const q2_wal_header_t* wal = (const q2_wal_header_t*)data;
  printf("  [WAL] %-32s  %4ux%-4u  mips: [%d, %d, %d, %d]  flags: 0x%04X\n",
         wal->name, wal->width, wal->height, wal->offsets[0], wal->offsets[1],
         wal->offsets[2], wal->offsets[3], wal->flags);
}

/* ========================================================================== *
 * Rendering Pipeline
 *
 * Implements the complete WebGPU rendering pipeline for Quake 2 BSP maps:
 * - Lightmap atlas building (BSP tree allocation, 4x brightness)
 * - WAL texture loading (palette-indexed → RGBA)
 * - BSP face triangulation with texture and lightmap UV computation
 * - Per-texture draw batching
 * - First-person camera with WASD movement
 * - ImGui debug overlay
 *
 * Port of go-quake2/render (OpenGL) to native WebGPU C99.
 * ========================================================================== */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations)
 * -------------------------------------------------------------------------- */

static const char* q2_vertex_shader_wgsl;
static const char* q2_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Lightmap Atlas BSP Tree Allocator
 *
 * Allocates rectangular regions on a 512x512 lightmap atlas using a binary
 * space partitioning tree. Matches the algorithm in go-quake2 lightmap.go.
 * -------------------------------------------------------------------------- */

#define LM_NODE_POOL_SIZE 16384

typedef struct {
  int32_t x, y, w, h;
  int32_t child_left;  /* Index into pool, -1 = no children */
  int32_t child_right; /* Index into pool, -1 = no children */
  bool filled;
} lm_node_t;

typedef struct {
  lm_node_t nodes[LM_NODE_POOL_SIZE];
  uint32_t count;
  uint8_t pixels[Q2_LIGHTMAP_ATLAS_SIZE * Q2_LIGHTMAP_ATLAS_SIZE * 4];
} lm_atlas_t;

static void lm_atlas_init(lm_atlas_t* atlas)
{
  memset(atlas->nodes, 0, sizeof(atlas->nodes));
  atlas->count    = 1;
  atlas->nodes[0] = (lm_node_t){
    .x           = 0,
    .y           = 0,
    .w           = Q2_LIGHTMAP_ATLAS_SIZE,
    .h           = Q2_LIGHTMAP_ATLAS_SIZE,
    .child_left  = -1,
    .child_right = -1,
    .filled      = false,
  };
  /* Initialize to white (fullbright) so non-lightmapped faces render at full
   * brightness when they sample from unallocated atlas regions */
  memset(atlas->pixels, 255, sizeof(atlas->pixels));
}

/**
 * @brief Navigate the lightmap BSP tree and find an empty spot of the right
 *        size. Returns node index on success, -1 on failure.
 */
static int32_t lm_allocate(lm_atlas_t* atlas, int32_t idx, int32_t w, int32_t h)
{
  lm_node_t* node = &atlas->nodes[idx];

  /* Check children if they exist */
  if (node->child_left >= 0) {
    int32_t r = lm_allocate(atlas, node->child_left, w, h);
    if (r >= 0) {
      return r;
    }
    return lm_allocate(atlas, node->child_right, w, h);
  }

  if (node->filled) {
    return -1;
  }
  if (node->w < w || node->h < h) {
    return -1;
  }

  /* Perfect fit */
  if (node->w == w && node->h == h) {
    node->filled = true;
    return idx;
  }

  /* Split - allocate 2 child nodes */
  if (atlas->count + 2 > LM_NODE_POOL_SIZE) {
    return -1;
  }

  int32_t left  = (int32_t)atlas->count++;
  int32_t right = (int32_t)atlas->count++;

  /* Re-fetch after pool growth (pointer may be stale) */
  node              = &atlas->nodes[idx];
  node->child_left  = left;
  node->child_right = right;

  /* Split along the dimension with largest excess */
  if ((node->w - w) > (node->h - h)) {
    atlas->nodes[left] = (lm_node_t){
      .x           = node->x,
      .y           = node->y,
      .w           = w,
      .h           = node->h,
      .child_left  = -1,
      .child_right = -1,
      .filled      = false,
    };
    atlas->nodes[right] = (lm_node_t){
      .x           = node->x + w,
      .y           = node->y,
      .w           = node->w - w,
      .h           = node->h,
      .child_left  = -1,
      .child_right = -1,
      .filled      = false,
    };
  }
  else {
    atlas->nodes[left] = (lm_node_t){
      .x           = node->x,
      .y           = node->y,
      .w           = node->w,
      .h           = h,
      .child_left  = -1,
      .child_right = -1,
      .filled      = false,
    };
    atlas->nodes[right] = (lm_node_t){
      .x           = node->x,
      .y           = node->y + h,
      .w           = node->w,
      .h           = node->h - h,
      .child_left  = -1,
      .child_right = -1,
      .filled      = false,
    };
  }

  return lm_allocate(atlas, left, w, h);
}

/**
 * @brief Copy a face's lightmap data into the atlas with 4x brightness
 *        scaling and proportional clamping (matches go-quake2 lightmap.go).
 */
static void lm_copy_face_data(lm_atlas_t* atlas, const lm_node_t* node,
                              const uint8_t* lm_data, uint32_t offset,
                              int32_t lm_w, int32_t lm_h)
{
  uint32_t src = offset;
  for (int32_t row = 0; row < lm_h; row++) {
    for (int32_t col = 0; col < lm_w; col++) {
      int32_t r = (int32_t)lm_data[src + 0] * 4;
      int32_t g = (int32_t)lm_data[src + 1] * 4;
      int32_t b = (int32_t)lm_data[src + 2] * 4;

      /* Rescale if any component exceeds 255 */
      int32_t mx = r > g ? r : g;
      if (b > mx) {
        mx = b;
      }
      if (mx > 255) {
        float t = 255.0f / (float)mx;
        r       = (int32_t)((float)r * t);
        g       = (int32_t)((float)g * t);
        b       = (int32_t)((float)b * t);
      }

      uint32_t dst = ((uint32_t)(node->y + row) * Q2_LIGHTMAP_ATLAS_SIZE
                      + (uint32_t)(node->x + col))
                     * 4;
      atlas->pixels[dst + 0] = (uint8_t)r;
      atlas->pixels[dst + 1] = (uint8_t)g;
      atlas->pixels[dst + 2] = (uint8_t)b;
      atlas->pixels[dst + 3] = 255;

      src += 3; /* BSP lightmap data is RGB (3 bytes per luxel) */
    }
  }
}

/* -------------------------------------------------------------------------- *
 * WAL Texture Decoder
 * -------------------------------------------------------------------------- */

/**
 * @brief Decode a Quake 2 WAL texture from palette-indexed to RGBA.
 *        Caller must free() the returned buffer.
 */
static uint8_t* q2_wal_decode(const uint8_t* data, uint32_t size,
                              uint32_t* out_w, uint32_t* out_h)
{
  if (size < sizeof(q2_wal_header_t)) {
    return NULL;
  }

  const q2_wal_header_t* hdr = (const q2_wal_header_t*)data;
  uint32_t w                 = hdr->width;
  uint32_t h                 = hdr->height;

  if (w == 0 || h == 0) {
    return NULL;
  }
  if ((uint32_t)hdr->offsets[0] + w * h > size) {
    return NULL;
  }

  uint8_t* rgba = (uint8_t*)malloc(w * h * 4);
  if (!rgba) {
    return NULL;
  }

  const uint8_t* indices = data + hdr->offsets[0];
  for (uint32_t i = 0; i < w * h; i++) {
    uint8_t pi      = indices[i];
    rgba[i * 4 + 0] = q2_palette[pi][0];
    rgba[i * 4 + 1] = q2_palette[pi][1];
    rgba[i * 4 + 2] = q2_palette[pi][2];
    rgba[i * 4 + 3] = 255;
  }

  *out_w = w;
  *out_h = h;
  return rgba;
}

/**
 * @brief Create a magenta/black checkerboard placeholder for missing textures.
 */
static uint8_t* q2_create_placeholder_texture(uint32_t* w, uint32_t* h)
{
  *w            = 64;
  *h            = 64;
  uint8_t* rgba = (uint8_t*)malloc(64 * 64 * 4);
  for (uint32_t y = 0; y < 64; y++) {
    for (uint32_t x = 0; x < 64; x++) {
      uint32_t idx  = (y * 64 + x) * 4;
      bool check    = ((x / 8) + (y / 8)) % 2;
      rgba[idx + 0] = check ? 255 : 0;
      rgba[idx + 1] = 0;
      rgba[idx + 2] = check ? 255 : 0;
      rgba[idx + 3] = 255;
    }
  }
  return rgba;
}

/* -------------------------------------------------------------------------- *
 * Geometry Types and Helpers
 * -------------------------------------------------------------------------- */

/* Render vertex: 7 floats = 28 bytes, matching go-quake2 TexturedVertex */
typedef struct {
  float pos[3];    /* World position (Y-up coordinate system) */
  float tex_uv[2]; /* Diffuse texture UV */
  float lm_uv[2];  /* Lightmap atlas UV */
} q2_render_vertex_t;

#define Q2_MAX_RENDER_TEXTURES 256

/* Per-texture draw batch */
typedef struct {
  char name[Q2_WAL_NAME_LEN];
  uint32_t wal_width;
  uint32_t wal_height;
  wgpu_texture_t gpu_tex;
  WGPUBindGroup bind_group;
  int32_t vert_offset; /* Start vertex in the VBO */
  int32_t vert_count;  /* Number of vertices */
} q2_tex_batch_t;

/* Per-face lightmap info (precomputed during atlas building) */
typedef struct {
  int32_t atlas_x, atlas_y; /* Allocated position in atlas */
  int32_t lm_w, lm_h;       /* Lightmap dimensions in luxels */
  float min_u, min_v;       /* Minimum texture-space UV */
  bool valid;               /* true if lightmap was allocated */
} q2_face_lm_t;

/* Per-face mesh data for PVS-based VBO rebuilds */
typedef struct {
  uint32_t start_vert; /* Start index in master_vertices[] */
  uint32_t vert_count; /* Number of vertices for this face */
  int32_t batch_idx;   /* Texture batch index (-1 = not renderable) */
} q2_face_mesh_t;

/* Pre-computed PVS visible face list per cluster */
typedef struct {
  uint32_t* faces;    /* Array of renderable face indices */
  uint32_t num_faces; /* Number of entries */
} q2_cluster_faces_t;

#define Q2_CLUSTER_INVALID 0xFFFF

/**
 * @brief Compute texture-space UV for a vertex (matches go-quake2
 * getTextureUV).
 */
static void q2_get_tex_uv(const q2_vertex_t* v, const q2_texinfo_t* ti,
                          float* u, float* vv)
{
  *u = v->x * ti->u_axis[0] + v->y * ti->u_axis[1] + v->z * ti->u_axis[2]
       + ti->u_offset;
  *vv = v->x * ti->v_axis[0] + v->y * ti->v_axis[1] + v->z * ti->v_axis[2]
        + ti->v_offset;
}

/**
 * @brief Compute lightmap dimensions for a face (matches go-quake2
 *        getLightmapDimensions). Returns width/height in luxels and the minimum
 *        texture-space UV coordinates.
 */
static void q2_get_lm_dims(const q2_bsp_map_t* map, const q2_face_t* face,
                           int32_t* w, int32_t* h, float* min_u, float* min_v)
{
  const q2_texinfo_t* ti = &map->texinfos[face->texinfo];

  /* Get first vertex */
  int32_t e0   = map->face_edges[face->first_edge];
  uint16_t vi0 = (e0 >= 0) ? map->edges[e0].v1 : map->edges[-e0].v2;

  float u0, v0;
  q2_get_tex_uv(&map->vertices[vi0], ti, &u0, &v0);

  double dmin_u = floor((double)u0), dmax_u = floor((double)u0);
  double dmin_v = floor((double)v0), dmax_v = floor((double)v0);

  for (uint16_t i = 1; i < face->num_edges; i++) {
    int32_t ei  = map->face_edges[face->first_edge + i];
    uint16_t vi = (ei >= 0) ? map->edges[ei].v1 : map->edges[-ei].v2;

    float u, v;
    q2_get_tex_uv(&map->vertices[vi], ti, &u, &v);

    double fu = floor((double)u), fv = floor((double)v);
    if (fu < dmin_u) {
      dmin_u = fu;
    }
    if (fv < dmin_v) {
      dmin_v = fv;
    }
    if (fu > dmax_u) {
      dmax_u = fu;
    }
    if (fv > dmax_v) {
      dmax_v = fv;
    }
  }

  *w     = (int32_t)(ceil(dmax_u / 16.0) - floor(dmin_u / 16.0) + 1);
  *h     = (int32_t)(ceil(dmax_v / 16.0) - floor(dmin_v / 16.0) + 1);
  *min_u = (float)floor(dmin_u);
  *min_v = (float)floor(dmin_v);
}

/**
 * @brief Extract vertex index from the BSP face-edge table.
 */
static uint16_t q2_face_vertex_idx(const q2_bsp_map_t* map,
                                   const q2_face_t* face, uint16_t idx)
{
  int32_t ei = map->face_edges[face->first_edge + idx];
  return (ei >= 0) ? map->edges[ei].v1 : map->edges[-ei].v2;
}

/* -------------------------------------------------------------------------- *
 * Entity String Parser
 * -------------------------------------------------------------------------- */

/**
 * @brief Find the first info_player_start entity and extract its origin and
 *        angle. Returns true on success.
 */
static bool q2_find_player_start(const char* entities, float out_pos[3],
                                 float* out_angle)
{
  const char* p = entities;
  while ((p = strstr(p, "info_player_start")) != NULL) {
    /* Walk back to find the containing '{' */
    const char* block = p;
    while (block > entities && *block != '{') {
      block--;
    }

    /* Find "origin" within this entity block */
    const char* origin = strstr(block, "\"origin\"");
    if (origin && origin < p + 256) {
      origin = strchr(origin + 8, '"');
      if (origin) {
        origin++;
        if (sscanf(origin, "%f %f %f", &out_pos[0], &out_pos[1], &out_pos[2])
            == 3) {
          *out_angle        = 0.0f;
          const char* angle = strstr(block, "\"angle\"");
          if (angle && angle < p + 256) {
            angle = strchr(angle + 7, '"');
            if (angle) {
              sscanf(angle + 1, "%f", out_angle);
            }
          }
          return true;
        }
      }
    }
    p++;
  }
  return false;
}

/* -------------------------------------------------------------------------- *
 * BSP Tree Traversal
 *
 * Walks the BSP tree to find which leaf contains a given position (in Q2
 * coordinate space). Returns the leaf index.
 * -------------------------------------------------------------------------- */

/**
 * @brief Find the BSP leaf node containing a world position (Q2 coordinates).
 *
 * Traverses from the root node (index 0) down the tree. Negative child
 * indices encode leaf nodes as -(leaf_index + 1).
 */
static int32_t q2_find_leaf(const q2_bsp_map_t* map, const float pos[3])
{
  int32_t node_id = 0;

  while (node_id >= 0) {
    const q2_node_t* node   = &map->nodes[node_id];
    const q2_plane_t* plane = &map->planes[node->plane];

    /* Signed distance to splitting plane */
    float d;
    if (plane->type < 3) {
      /* Axis-aligned plane (fast path) */
      d = pos[plane->type] - plane->dist;
    }
    else {
      /* Arbitrary plane (dot product) */
      d = pos[0] * plane->normal[0] + pos[1] * plane->normal[1]
          + pos[2] * plane->normal[2] - plane->dist;
    }

    node_id = (d >= 0.0f) ? node->front_child : node->back_child;
  }

  return -(node_id + 1);
}

/* -------------------------------------------------------------------------- *
 * PVS (Potentially Visible Set) Pre-computation
 *
 * For each cluster, decompresses the RLE-encoded PVS bit vector and collects
 * all renderable faces visible from that cluster. Stored at load time so the
 * per-frame cost is just a BSP traversal + pointer swap.
 * -------------------------------------------------------------------------- */

/**
 * @brief Decompress PVS for a given cluster and collect visible face indices.
 *
 * @param map         Parsed BSP data.
 * @param cluster     Source cluster ID.
 * @param face_meshes Per-face mesh info (to check renderability).
 * @param out_faces   Output array of face indices (caller must free).
 * @param out_count   Output count.
 */
static void q2_decompress_pvs(const q2_bsp_map_t* map, uint32_t cluster,
                              const q2_face_mesh_t* face_meshes,
                              uint32_t** out_faces, uint32_t* out_count)
{
  uint32_t num_clusters = map->num_vis_clusters;

  /* Temporary visibility byte array (one byte per cluster) */
  uint8_t* vis = (uint8_t*)calloc(num_clusters, 1);
  if (!vis) {
    *out_faces = NULL;
    *out_count = 0;
    return;
  }

  /* A cluster is always visible from itself */
  vis[cluster] = 1;

  /* Decompress the RLE-encoded PVS */
  if (map->vis_offsets && map->vis_data) {
    uint32_t v = map->vis_offsets[cluster].pvs;
    uint32_t c = 0;
    /* The pvs offset is relative to the start of the visibility lump.
     * vis_offsets was parsed at (lump_start + 4), so lump_start is 4 bytes
     * before vis_offsets. Use direct pointer math to reach lump start. */
    const uint8_t* lump_base
      = (const uint8_t*)map->vis_offsets - sizeof(uint32_t);
    /* Now pvs byte is simply at lump_base[v] */

    while (c < num_clusters) {
      uint8_t byte = lump_base[v];
      if (byte == 0) {
        /* RLE: skip clusters */
        v++;
        c += 8 * (uint32_t)lump_base[v];
      }
      else {
        /* 8 visibility bits */
        for (uint8_t bit = 0; bit < 8 && c < num_clusters; bit++, c++) {
          if (byte & (1u << bit)) {
            vis[c] = 1;
          }
        }
      }
      v++;
    }
  }

  /* Collect all renderable faces from visible clusters */
  /* Build cluster → leaf → face mapping */
  /* First, count faces to allocate output array */
  uint32_t capacity = 0;
  for (uint32_t li = 0; li < map->num_leaves; li++) {
    uint16_t lc = map->leaves[li].cluster;
    if (lc == Q2_CLUSTER_INVALID || lc >= num_clusters || !vis[lc]) {
      continue;
    }
    capacity += map->leaves[li].num_leaf_faces;
  }

  uint32_t* faces = (uint32_t*)malloc(capacity * sizeof(uint32_t));
  uint32_t count  = 0;

  /* Use a bitfield to deduplicate face indices */
  uint32_t bitmask_size = (map->num_faces + 31) / 32;
  uint32_t* seen        = (uint32_t*)calloc(bitmask_size, sizeof(uint32_t));

  for (uint32_t li = 0; li < map->num_leaves; li++) {
    const q2_leaf_t* leaf = &map->leaves[li];
    if (leaf->cluster == Q2_CLUSTER_INVALID || leaf->cluster >= num_clusters
        || !vis[leaf->cluster]) {
      continue;
    }

    for (uint16_t fi = 0; fi < leaf->num_leaf_faces; fi++) {
      uint32_t face_idx = map->leaf_faces[leaf->first_leaf_face + fi];
      if (face_idx >= map->num_faces) {
        continue;
      }

      /* Deduplicate */
      uint32_t word = face_idx / 32;
      uint32_t bit  = face_idx % 32;
      if (seen[word] & (1u << bit)) {
        continue;
      }
      seen[word] |= (1u << bit);

      /* Only include faces that are renderable */
      if (face_meshes[face_idx].batch_idx >= 0) {
        faces[count++] = face_idx;
      }
    }
  }

  free(seen);
  free(vis);

  *out_faces = faces;
  *out_count = count;
}

/**
 * @brief Pre-compute PVS face lists for all clusters.
 *        Call once after geometry is built.
 */
static q2_cluster_faces_t*
q2_precompute_all_pvs(const q2_bsp_map_t* map,
                      const q2_face_mesh_t* face_meshes,
                      uint32_t* out_num_clusters)
{
  uint32_t num_clusters = map->num_vis_clusters;
  *out_num_clusters     = num_clusters;

  if (num_clusters == 0) {
    return NULL;
  }

  q2_cluster_faces_t* clusters
    = (q2_cluster_faces_t*)calloc(num_clusters, sizeof(q2_cluster_faces_t));

  for (uint32_t c = 0; c < num_clusters; c++) {
    q2_decompress_pvs(map, c, face_meshes, &clusters[c].faces,
                      &clusters[c].num_faces);
  }

  printf("[Q2] PVS pre-computed for %u clusters\n", num_clusters);
  return clusters;
}

/**
 * @brief Free pre-computed PVS data.
 */
static void q2_cluster_faces_destroy(q2_cluster_faces_t* clusters,
                                     uint32_t num_clusters)
{
  if (clusters) {
    for (uint32_t c = 0; c < num_clusters; c++) {
      free(clusters[c].faces);
    }
    free(clusters);
  }
}

/* -------------------------------------------------------------------------- *
 * Renderer State
 * -------------------------------------------------------------------------- */

static struct {
  /* Source data */
  q2_pak_t pak;
  q2_bsp_map_t bsp;

  /* Camera */
  camera_t camera;

  /* Timing */
  uint64_t last_frame_time;

  /* Texture batches */
  q2_tex_batch_t textures[Q2_MAX_RENDER_TEXTURES];
  uint32_t num_textures;

  /* Vertex count */
  uint32_t total_vertices;

  /* Per-face mesh data for PVS-based VBO rebuilds */
  q2_render_vertex_t* master_vertices; /* All vertex data (retained) */
  uint32_t master_vert_count;
  q2_face_mesh_t* face_meshes; /* Per-face VBO metadata (num_faces entries) */

  /* Pre-computed PVS per cluster */
  q2_cluster_faces_t* cluster_faces;
  uint32_t num_pvs_clusters;

  /* BSP leaf tracking (for PVS updates) */
  int32_t current_leaf;
  int32_t prev_leaf;

  /* Lightmap atlas GPU texture */
  wgpu_texture_t lightmap_tex;

  /* GPU resources */
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t uniform_buffer;
  WGPUBindGroupLayout bg_layout_shared;
  WGPUBindGroupLayout bg_layout_texture;
  WGPUBindGroup bg_shared;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUSampler sampler;

  /* Render pass */
  WGPURenderPassColorAttachment color_att;
  WGPURenderPassDepthStencilAttachment depth_att;
  WGPURenderPassDescriptor rp_desc;

  WGPUBool initialized;
} state = {
  .current_leaf = -1,
  .prev_leaf    = -2, /* Force initial rebuild */
  .color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.05, 0.05, 0.08, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .rp_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_att,
    .depthStencilAttachment = &state.depth_att,
  },
};

/* -------------------------------------------------------------------------- *
 * Map Geometry Builder
 *
 * Extracts triangulated geometry from BSP faces, computes texture and lightmap
 * UVs, builds per-texture draw batches, and populates the lightmap atlas.
 * -------------------------------------------------------------------------- */

static q2_render_vertex_t* build_map_geometry(q2_bsp_map_t* map, q2_pak_t* pak,
                                              lm_atlas_t* atlas,
                                              uint32_t* out_total_verts)
{
  /* --- Pass 1: Build unique texture list --- */
  uint32_t tex_count = 0;
  int32_t texinfo_to_batch[Q2_MAX_MAP_TEXTURES];
  memset(texinfo_to_batch, -1, sizeof(texinfo_to_batch));

  for (uint32_t i = 0; i < map->num_texinfos && i < Q2_MAX_MAP_TEXTURES; i++) {
    const char* name = map->texinfos[i].texture_name;
    if (name[0] == '\0') {
      continue;
    }

    /* Check if already in batch list */
    int32_t found = -1;
    for (uint32_t j = 0; j < tex_count; j++) {
      if (strncmp(state.textures[j].name, name, Q2_WAL_NAME_LEN) == 0) {
        found = (int32_t)j;
        break;
      }
    }

    if (found >= 0) {
      texinfo_to_batch[i] = found;
    }
    else if (tex_count < Q2_MAX_RENDER_TEXTURES) {
      strncpy(state.textures[tex_count].name, name, Q2_WAL_NAME_LEN - 1);
      state.textures[tex_count].name[Q2_WAL_NAME_LEN - 1] = '\0';
      state.textures[tex_count].wal_width                 = 64;
      state.textures[tex_count].wal_height                = 64;
      state.textures[tex_count].vert_offset               = 0;
      state.textures[tex_count].vert_count                = 0;

      /* Get WAL texture dimensions from PAK for UV normalization */
      char tex_path[128];
      snprintf(tex_path, sizeof(tex_path), "textures/%s.wal", name);
      const q2_pak_entry_t* entry = q2_pak_find(pak, tex_path);
      if (entry) {
        uint32_t wal_size       = 0;
        const uint8_t* wal_data = q2_pak_get_data(pak, entry, &wal_size);
        if (wal_data && wal_size >= sizeof(q2_wal_header_t)) {
          const q2_wal_header_t* hdr = (const q2_wal_header_t*)wal_data;
          if (hdr->width > 0 && hdr->height > 0) {
            state.textures[tex_count].wal_width  = hdr->width;
            state.textures[tex_count].wal_height = hdr->height;
          }
        }
      }

      texinfo_to_batch[i] = (int32_t)tex_count;
      tex_count++;
    }
  }
  state.num_textures = tex_count;

  /* --- Pass 2: Allocate lightmaps in atlas for each face --- */
  q2_face_lm_t* face_lm
    = (q2_face_lm_t*)calloc(map->num_faces, sizeof(q2_face_lm_t));

  for (uint32_t fi = 0; fi < map->num_faces; fi++) {
    const q2_face_t* face = &map->faces[fi];
    if (face->num_edges < 3) {
      continue;
    }

    uint32_t flags = 0;
    if (face->texinfo < map->num_texinfos) {
      flags = map->texinfos[face->texinfo].flags;
    }

    /* Skip non-renderable and non-lightmapped faces */
    if (flags & (Q2_SURF_SKY | Q2_SURF_NODRAW | Q2_SURF_WARP)) {
      continue;
    }
    if (face->lightmap_styles[0] == 0xFF) {
      continue;
    }

    /* Compute lightmap dimensions */
    int32_t lm_w, lm_h;
    float min_u, min_v;
    q2_get_lm_dims(map, face, &lm_w, &lm_h, &min_u, &min_v);

    if (lm_w <= 0 || lm_h <= 0 || lm_w > 256 || lm_h > 256) {
      continue;
    }

    /* Allocate in atlas */
    int32_t node_idx = lm_allocate(atlas, 0, lm_w, lm_h);
    if (node_idx < 0) {
      continue;
    }

    const lm_node_t* node = &atlas->nodes[node_idx];

    /* Copy lightmap pixel data with brightness scaling */
    if (face->lightmap_offset < map->num_lightmap_bytes) {
      lm_copy_face_data(atlas, node, map->lightmap_data, face->lightmap_offset,
                        lm_w, lm_h);
    }

    face_lm[fi].atlas_x = node->x;
    face_lm[fi].atlas_y = node->y;
    face_lm[fi].lm_w    = lm_w;
    face_lm[fi].lm_h    = lm_h;
    face_lm[fi].min_u   = min_u;
    face_lm[fi].min_v   = min_v;
    face_lm[fi].valid   = true;
  }

  /* --- Pass 3: Count triangulated vertices and build per-face metadata --- */
  state.face_meshes
    = (q2_face_mesh_t*)calloc(map->num_faces, sizeof(q2_face_mesh_t));
  /* Initialize all faces as non-renderable */
  for (uint32_t fi = 0; fi < map->num_faces; fi++) {
    state.face_meshes[fi].batch_idx = -1;
  }

  uint32_t total_verts = 0;
  for (uint32_t fi = 0; fi < map->num_faces; fi++) {
    const q2_face_t* face = &map->faces[fi];
    if (face->num_edges < 3) {
      continue;
    }

    uint32_t flags = 0;
    if (face->texinfo < map->num_texinfos) {
      flags = map->texinfos[face->texinfo].flags;
    }
    if (flags & (Q2_SURF_SKY | Q2_SURF_NODRAW)) {
      continue;
    }

    int32_t batch = (face->texinfo < Q2_MAX_MAP_TEXTURES) ?
                      texinfo_to_batch[face->texinfo] :
                      -1;
    if (batch < 0) {
      continue;
    }

    uint32_t tri_verts               = (face->num_edges - 2) * 3;
    state.face_meshes[fi].vert_count = tri_verts;
    state.face_meshes[fi].batch_idx  = batch;
    state.textures[batch].vert_count += (int32_t)tri_verts;
    total_verts += tri_verts;
  }

  /* Compute per-texture vertex offsets (prefix sum) — these will be
   * recalculated by rebuild_visible_geometry() but we need the total count
   * for the master vertex array allocation. */
  uint32_t _offset = 0;
  for (uint32_t t = 0; t < tex_count; t++) {
    _offset += (uint32_t)state.textures[t].vert_count;
    state.textures[t].vert_count = 0; /* Reset for use as write cursor */
  }
  (void)_offset;

  /* --- Pass 4: Generate triangulated vertices and record per-face offsets ---
   */
  q2_render_vertex_t* verts
    = (q2_render_vertex_t*)calloc(total_verts, sizeof(q2_render_vertex_t));

  /* Global write cursor tracks absolute vertex positions */
  uint32_t global_write = 0;

  for (uint32_t fi = 0; fi < map->num_faces; fi++) {
    const q2_face_t* face = &map->faces[fi];
    if (face->num_edges < 3) {
      continue;
    }

    int32_t batch = state.face_meshes[fi].batch_idx;
    if (batch < 0) {
      continue;
    }

    uint32_t flags = 0;
    if (face->texinfo < map->num_texinfos) {
      flags = map->texinfos[face->texinfo].flags;
    }
    if (flags & (Q2_SURF_SKY | Q2_SURF_NODRAW)) {
      continue;
    }

    const q2_texinfo_t* ti = &map->texinfos[face->texinfo];
    float tw               = (float)state.textures[batch].wal_width;
    float th               = (float)state.textures[batch].wal_height;

    /* Record the start vertex for this face in the master array */
    state.face_meshes[fi].start_vert = global_write;

    /* First vertex of the fan */
    uint16_t vi0          = q2_face_vertex_idx(map, face, 0);
    const q2_vertex_t* v0 = &map->vertices[vi0];

    /* Triangle fan triangulation: (v0, v[i-1], v[i]) for i=2..N-1 */
    for (uint16_t i = 2; i < face->num_edges; i++) {
      uint16_t vi1          = q2_face_vertex_idx(map, face, i - 1);
      uint16_t vi2          = q2_face_vertex_idx(map, face, i);
      const q2_vertex_t* v1 = &map->vertices[vi1];
      const q2_vertex_t* v2 = &map->vertices[vi2];

      const q2_vertex_t* tri[3] = {v0, v1, v2};

      for (int k = 0; k < 3; k++) {
        q2_render_vertex_t* rv = &verts[global_write + k];
        const q2_vertex_t* sv  = tri[k];

        /* Coordinate conversion: Q2 (X right, Y forward, Z up)
         *                      → WebGPU (X right, Y up, -Z forward) */
        rv->pos[0] = sv->x;
        rv->pos[1] = sv->z;
        rv->pos[2] = -sv->y;

        /* Diffuse texture UV (normalized by texture dimensions) */
        float raw_u, raw_v;
        q2_get_tex_uv(sv, ti, &raw_u, &raw_v);
        rv->tex_uv[0] = raw_u / tw;
        rv->tex_uv[1] = raw_v / th;

        /* Lightmap atlas UV */
        if (face_lm[fi].valid) {
          float s = raw_u - face_lm[fi].min_u;
          s += (float)(face_lm[fi].atlas_x * 16 + 8);
          s /= (float)(Q2_LIGHTMAP_ATLAS_SIZE * 16);

          float t = raw_v - face_lm[fi].min_v;
          t += (float)(face_lm[fi].atlas_y * 16 + 8);
          t /= (float)(Q2_LIGHTMAP_ATLAS_SIZE * 16);

          rv->lm_uv[0] = s;
          rv->lm_uv[1] = t;
        }
        else {
          /* No lightmap - sample from unallocated white region */
          rv->lm_uv[0] = 0.999f;
          rv->lm_uv[1] = 0.999f;
        }
      }

      global_write += 3;
    }
  }

  free(face_lm);
  state.master_vertices   = verts;
  state.master_vert_count = total_verts;
  *out_total_verts        = total_verts;
  return verts;
}

/* -------------------------------------------------------------------------- *
 * PVS-Based VBO Rebuild
 *
 * When the camera enters a new BSP leaf, rebuilds a compact vertex buffer
 * containing only the visible faces (sorted by texture batch) and updates
 * per-batch draw ranges.
 * -------------------------------------------------------------------------- */

/**
 * @brief Rebuild VBO from a set of visible face indices.
 *
 * Copies vertex data from the master array into a compact buffer sorted by
 * texture batch, then uploads it to the GPU vertex buffer.
 *
 * @param wgpu_context  WebGPU context for buffer upload.
 * @param vis_faces     Array of visible face indices.
 * @param num_vis_faces Number of entries.
 */
static void rebuild_visible_geometry(wgpu_context_t* wgpu_context,
                                     const uint32_t* vis_faces,
                                     uint32_t num_vis_faces)
{
  /* Reset all batch counts */
  for (uint32_t t = 0; t < state.num_textures; t++) {
    state.textures[t].vert_offset = 0;
    state.textures[t].vert_count  = 0;
  }

  /* Count visible vertices per batch */
  uint32_t total_vis_verts = 0;
  for (uint32_t i = 0; i < num_vis_faces; i++) {
    uint32_t fi   = vis_faces[i];
    int32_t batch = state.face_meshes[fi].batch_idx;
    uint32_t nv   = state.face_meshes[fi].vert_count;
    if (batch >= 0 && nv > 0) {
      state.textures[batch].vert_count += (int32_t)nv;
      total_vis_verts += nv;
    }
  }

  /* Prefix sum for batch offsets */
  int32_t offset = 0;
  for (uint32_t t = 0; t < state.num_textures; t++) {
    state.textures[t].vert_offset = offset;
    offset += state.textures[t].vert_count;
    state.textures[t].vert_count = 0; /* Reset for write cursor */
  }

  /* Allocate temporary compact buffer */
  if (total_vis_verts == 0) {
    state.total_vertices = 0;
    return;
  }

  q2_render_vertex_t* compact
    = (q2_render_vertex_t*)malloc(total_vis_verts * sizeof(q2_render_vertex_t));

  /* Copy visible face vertices into batch-sorted positions */
  for (uint32_t i = 0; i < num_vis_faces; i++) {
    uint32_t fi   = vis_faces[i];
    int32_t batch = state.face_meshes[fi].batch_idx;
    uint32_t nv   = state.face_meshes[fi].vert_count;
    if (batch < 0 || nv == 0) {
      continue;
    }

    uint32_t dst = (uint32_t)state.textures[batch].vert_offset
                   + (uint32_t)state.textures[batch].vert_count;
    memcpy(&compact[dst],
           &state.master_vertices[state.face_meshes[fi].start_vert],
           nv * sizeof(q2_render_vertex_t));
    state.textures[batch].vert_count += (int32_t)nv;
  }

  state.total_vertices = total_vis_verts;

  /* Upload to GPU */
  uint64_t buf_size = total_vis_verts * sizeof(q2_render_vertex_t);

  /* If the existing buffer is large enough, just write; otherwise recreate */
  if (state.vertex_buffer.buffer && state.vertex_buffer.size >= buf_size) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.vertex_buffer.buffer, 0,
                         compact, buf_size);
  }
  else {
    if (state.vertex_buffer.buffer) {
      WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer.buffer)
    }
    state.vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Q2 vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = buf_size,
                      .initial.data = compact,
                    });
  }

  free(compact);
}

/* -------------------------------------------------------------------------- *
 * Map Data Loading
 * -------------------------------------------------------------------------- */

static bool load_map_data(void)
{
  const char* pak_path
    = getenv("Q2_PAK_PATH") ?
        getenv("Q2_PAK_PATH") :
        "/home/sdauwe/GitHub/quake2generic/build/baseq2/pak0.pak";

  if (!q2_pak_load(pak_path, &state.pak)) {
    fprintf(stderr, "[ERROR] Could not load PAK: %s\n", pak_path);
    return false;
  }

  /* Find a BSP map */
  static const char* known_maps[] = {
    "maps/demo1.bsp",
    "maps/demo2.bsp",
    "maps/base1.bsp",
  };

  const q2_pak_entry_t* bsp_entry = NULL;
  for (uint32_t i = 0;
       i < sizeof(known_maps) / sizeof(known_maps[0]) && !bsp_entry; i++) {
    bsp_entry = q2_pak_find(&state.pak, known_maps[i]);
  }
  if (!bsp_entry) {
    for (uint32_t i = 0; i < state.pak.num_entries; i++) {
      const char* name = state.pak.entries[i].filename;
      uint32_t len     = (uint32_t)strnlen(name, Q2_PAK_FILENAME_LEN);
      if (len > 4 && strncmp(name + len - 4, ".bsp", 4) == 0) {
        bsp_entry = &state.pak.entries[i];
        break;
      }
    }
  }

  if (!bsp_entry) {
    fprintf(stderr, "[ERROR] No BSP files found in PAK\n");
    return false;
  }

  uint32_t bsp_size       = 0;
  const uint8_t* bsp_data = q2_pak_get_data(&state.pak, bsp_entry, &bsp_size);
  if (!bsp_data || !q2_bsp_parse(bsp_data, bsp_size, &state.bsp)) {
    fprintf(stderr, "[ERROR] BSP parsing failed\n");
    return false;
  }

  printf("[Q2] Loaded BSP: %s (%u faces, %u texinfos, %u vertices)\n",
         bsp_entry->filename, state.bsp.num_faces, state.bsp.num_texinfos,
         state.bsp.num_vertices);
  return true;
}

/* -------------------------------------------------------------------------- *
 * GPU Resource Initialization
 * -------------------------------------------------------------------------- */

static void init_gpu_textures(wgpu_context_t* wgpu_context)
{
  uint32_t loaded = 0, missing = 0;

  for (uint32_t t = 0; t < state.num_textures; t++) {
    char tex_path[128];
    snprintf(tex_path, sizeof(tex_path), "textures/%s.wal",
             state.textures[t].name);

    const q2_pak_entry_t* entry = q2_pak_find(&state.pak, tex_path);
    uint32_t w = 64, h = 64;
    uint8_t* rgba = NULL;

    if (entry) {
      uint32_t wal_size       = 0;
      const uint8_t* wal_data = q2_pak_get_data(&state.pak, entry, &wal_size);
      if (wal_data) {
        rgba = q2_wal_decode(wal_data, wal_size, &w, &h);
      }
    }

    if (rgba) {
      loaded++;
    }
    else {
      rgba = q2_create_placeholder_texture(&w, &h);
      missing++;
    }

    state.textures[t].gpu_tex = wgpu_create_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .extent
        = (WGPUExtent3D){.width = w, .height = h, .depthOrArrayLayers = 1},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .pixels = {.ptr = rgba, .size = w * h * 4},
      });

    free(rgba);
  }

  printf("[Q2] Textures: %u loaded, %u missing (placeholder)\n", loaded,
         missing);
}

static void init_sampler(wgpu_context_t* wgpu_context)
{
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Q2 texture sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Group 0: Shared (MVP uniform + sampler + lightmap texture) */
  WGPUBindGroupLayoutEntry shared_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(mat4),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  state.bg_layout_shared = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Q2 shared bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(shared_entries),
                            .entries    = shared_entries,
                          });
  ASSERT(state.bg_layout_shared != NULL);

  /* Group 1: Per-texture (diffuse texture only) */
  WGPUBindGroupLayoutEntry tex_entry = (WGPUBindGroupLayoutEntry){
    .binding    = 0,
    .visibility = WGPUShaderStage_Fragment,
    .texture = (WGPUTextureBindingLayout){
      .sampleType    = WGPUTextureSampleType_Float,
      .viewDimension = WGPUTextureViewDimension_2D,
      .multisampled  = false,
    },
    .storageTexture = {0},
  };
  state.bg_layout_texture = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Q2 texture bind group layout"),
                            .entryCount = 1,
                            .entries    = &tex_entry,
                          });
  ASSERT(state.bg_layout_texture != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Shared bind group (group 0) */
  WGPUBindGroupEntry shared_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.uniform_buffer.buffer,
      .offset  = 0,
      .size    = sizeof(mat4),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.lightmap_tex.view,
    },
  };
  state.bg_shared = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Q2 shared bind group"),
                            .layout     = state.bg_layout_shared,
                            .entryCount = (uint32_t)ARRAY_SIZE(shared_entries),
                            .entries    = shared_entries,
                          });
  ASSERT(state.bg_shared != NULL);

  /* Per-texture bind groups (group 1) */
  for (uint32_t t = 0; t < state.num_textures; t++) {
    WGPUBindGroupEntry tex_entry = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.textures[t].gpu_tex.view,
    };
    state.textures[t].bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Q2 texture bind group"),
                              .layout     = state.bg_layout_texture,
                              .entryCount = 1,
                              .entries    = &tex_entry,
                            });
    ASSERT(state.textures[t].bind_group != NULL);
  }
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout with two bind group layouts */
  WGPUBindGroupLayout layouts[2]
    = {state.bg_layout_shared, state.bg_layout_texture};
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Q2 pipeline layout"),
                            .bindGroupLayoutCount = 2,
                            .bindGroupLayouts     = layouts,
                          });
  ASSERT(state.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule vs
    = wgpu_create_shader_module(wgpu_context->device, q2_vertex_shader_wgsl);
  WGPUShaderModule fs
    = wgpu_create_shader_module(wgpu_context->device, q2_fragment_shader_wgsl);

  /* Blend and depth stencil states */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  WGPUDepthStencilState depth_stencil
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout: pos(3) + tex_uv(2) + lm_uv(2) */
  WGPU_VERTEX_BUFFER_LAYOUT(
    q2_vertex, sizeof(q2_render_vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       offsetof(q2_render_vertex_t, pos)),
    /* Attribute location 1: Texture UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                       offsetof(q2_render_vertex_t, tex_uv)),
    /* Attribute location 2: Lightmap UV */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       offsetof(q2_render_vertex_t, lm_uv)))

  /* Render pipeline */
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Q2 render pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = vs,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &q2_vertex_vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fs,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &depth_stencil,
      .multisample  = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vs);
  wgpuShaderModuleRelease(fs);
}

/* -------------------------------------------------------------------------- *
 * Camera Setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type   = CameraType_FirstPerson;
  state.camera.flip_y = true;

  /* Try to get spawn position from entity string */
  float spawn_pos[3] = {0};
  float spawn_angle  = 0.0f;

  if (state.bsp.entities
      && q2_find_player_start(state.bsp.entities, spawn_pos, &spawn_angle)) {
    /* Convert Q2 coords (X right, Y fwd, Z up) to render (X right, Y up, -Z
     * fwd). With flip_y=true the view matrix negates Y internally, so we store
     * the Y component with its natural sign: position = (-wx, +wy, -wz). */
    camera_set_position(&state.camera,
                        (vec3){-spawn_pos[0], spawn_pos[2], spawn_pos[1]});
    camera_set_rotation(&state.camera,
                        (vec3){0.0f, -(spawn_angle - 90.0f), 0.0f});
    printf("[Q2] Spawn: (%.0f, %.0f, %.0f) angle=%.0f\n", spawn_pos[0],
           spawn_pos[1], spawn_pos[2], spawn_angle);
  }
  else if (state.bsp.num_models > 0) {
    const q2_model_t* world = &state.bsp.models[0];
    float cx                = (world->mins[0] + world->maxs[0]) * 0.5f;
    float cy                = (world->mins[1] + world->maxs[1]) * 0.5f;
    float cz                = (world->mins[2] + world->maxs[2]) * 0.5f;
    camera_set_position(&state.camera, (vec3){-cx, cz, cy});
    printf("[Q2] Camera at world center: (%.0f, %.0f, %.0f)\n", cx, cy, cz);
  }

  camera_set_perspective(
    &state.camera, 70.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 10000.0f);
  camera_set_movement_speed(&state.camera, 400.0f);
  camera_set_rotation_speed(&state.camera, 0.25f);
}

/* -------------------------------------------------------------------------- *
 * ImGui Overlay
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Quake 2 Renderer", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  igText("Faces: %u / %u (visible/total)", state.total_vertices / 3,
         state.master_vert_count / 3);
  igText("Vertices: %u", state.total_vertices);
  igText("Textures: %u", state.num_textures);
  igSeparator();
  igText("BSP Leaf: %d", state.current_leaf);
  if (state.current_leaf >= 0
      && (uint32_t)state.current_leaf < state.bsp.num_leaves) {
    uint16_t cluster = state.bsp.leaves[state.current_leaf].cluster;
    igText("Cluster: %u", (uint32_t)cluster);
  }
  igSeparator();
  igText("Camera: (%.0f, %.0f, %.0f)", -state.camera.position[0],
         state.camera.position[1], -state.camera.position[2]);
  igText("Rotation: (%.1f, %.1f)", state.camera.rotation[0],
         state.camera.rotation[1]);
  igSeparator();
  igText("WASD = move, Mouse = look");

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * WebGPU Callbacks
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  /* Load PAK and BSP */
  if (!load_map_data()) {
    return EXIT_FAILURE;
  }

  /* Build lightmap atlas (heap-allocated, ~1.4 MB) */
  lm_atlas_t* atlas = (lm_atlas_t*)calloc(1, sizeof(lm_atlas_t));
  if (!atlas) {
    return EXIT_FAILURE;
  }
  lm_atlas_init(atlas);

  /* Build triangulated geometry with UVs (stored as master vertex array) */
  q2_render_vertex_t* verts
    = build_map_geometry(&state.bsp, &state.pak, atlas, &state.total_vertices);
  if (!verts || state.total_vertices == 0) {
    free(atlas);
    return EXIT_FAILURE;
  }

  printf("[Q2] Geometry: %u master vertices (%.1f KB)\n",
         state.master_vert_count,
         state.master_vert_count * sizeof(q2_render_vertex_t) / 1024.0);

  /* Pre-compute PVS face lists for all clusters */
  state.cluster_faces = q2_precompute_all_pvs(&state.bsp, state.face_meshes,
                                              &state.num_pvs_clusters);

  /* Initial VBO upload: use all vertices (will be replaced by PVS on first
   * leaf detection). Allocate with generous capacity for the maximum possible
   * visible set to avoid re-creating the buffer on every leaf change. */
  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Q2 vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = state.master_vert_count * sizeof(q2_render_vertex_t),
      .initial.data = verts,
    });
  /* Note: do NOT free verts - it's stored as state.master_vertices */

  /* Initial rebuild: sort all renderable faces into batch order so the draw
   * calls have correct vert_offset/vert_count from the first frame. */
  {
    uint32_t* all_faces
      = (uint32_t*)malloc(state.bsp.num_faces * sizeof(uint32_t));
    uint32_t num_all = 0;
    for (uint32_t fi = 0; fi < state.bsp.num_faces; fi++) {
      if (state.face_meshes[fi].batch_idx >= 0) {
        all_faces[num_all++] = fi;
      }
    }
    if (num_all > 0) {
      rebuild_visible_geometry(wgpu_context, all_faces, num_all);
    }
    free(all_faces);
  }

  /* Upload lightmap atlas texture to GPU */
  state.lightmap_tex = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = Q2_LIGHTMAP_ATLAS_SIZE,
        .height             = Q2_LIGHTMAP_ATLAS_SIZE,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = atlas->pixels,
        .size = Q2_LIGHTMAP_ATLAS_SIZE * Q2_LIGHTMAP_ATLAS_SIZE * 4,
      },
    });
  printf("[Q2] Lightmap atlas: %dx%d (%u nodes used)\n", Q2_LIGHTMAP_ATLAS_SIZE,
         Q2_LIGHTMAP_ATLAS_SIZE, atlas->count);
  free(atlas);

  /* Decode and upload WAL textures */
  init_gpu_textures(wgpu_context);

  /* Uniform buffer (MVP matrix) */
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4),
                  });

  /* GPU pipeline setup */
  init_sampler(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_bind_groups(wgpu_context);
  init_pipeline(wgpu_context);

  /* Camera */
  init_camera(wgpu_context);

  /* ImGui overlay */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  printf("[Q2] Renderer initialized successfully\n");
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Delta time */
  uint64_t now          = stm_now();
  float dt              = (state.last_frame_time == 0) ?
                            (1.0f / 60.0f) :
                            (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Update camera */
  camera_update(&state.camera, dt);

  /* --- BSP traversal: find current leaf and update VBO via PVS --- */
  if (state.cluster_faces && state.num_pvs_clusters > 0) {
    /* Convert camera position to Q2 coordinate space for BSP traversal.
     * With flip_y=true: world=(−pos.x, pos.y, −pos.z)
     * Q2 coords: x=world_x, y=−world_z, z=world_y */
    float q2_pos[3] = {
      -state.camera.position[0], /* Q2 X = render X */
      state.camera.position[2],  /* Q2 Y = −render Z */
      state.camera.position[1],  /* Q2 Z = render Y */
    };

    int32_t leaf_idx   = q2_find_leaf(&state.bsp, q2_pos);
    state.current_leaf = leaf_idx;

    /* Rebuild VBO only when entering a new leaf */
    if (leaf_idx != state.prev_leaf && leaf_idx >= 0
        && (uint32_t)leaf_idx < state.bsp.num_leaves) {
      uint16_t cluster = state.bsp.leaves[leaf_idx].cluster;
      if (cluster != Q2_CLUSTER_INVALID && cluster < state.num_pvs_clusters) {
        const q2_cluster_faces_t* cf = &state.cluster_faces[cluster];
        if (cf->num_faces > 0) {
          rebuild_visible_geometry(wgpu_context, cf->faces, cf->num_faces);
        }
      }
      state.prev_leaf = leaf_idx;
    }
  }

  /* Compute and upload MVP matrix */
  mat4 mvp;
  glm_mat4_mul(state.camera.matrices.perspective, state.camera.matrices.view,
               mvp);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &mvp, sizeof(mat4));

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* Update render pass attachments */
  state.color_att.view = wgpu_context->swapchain_view;
  state.depth_att.view = wgpu_context->depth_stencil_view;

  /* Create command encoder and render pass */
  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rp
    = wgpuCommandEncoderBeginRenderPass(enc, &state.rp_desc);

  /* Bind pipeline and shared resources */
  wgpuRenderPassEncoderSetPipeline(rp, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.vertex_buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bg_shared, 0, 0);

  /* Draw per-texture batches */
  for (uint32_t t = 0; t < state.num_textures; t++) {
    if (state.textures[t].vert_count <= 0) {
      continue;
    }
    wgpuRenderPassEncoderSetBindGroup(rp, 1, state.textures[t].bind_group, 0,
                                      0);
    wgpuRenderPassEncoderDraw(rp, (uint32_t)state.textures[t].vert_count, 1,
                              (uint32_t)state.textures[t].vert_offset, 0);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(rp);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);

  /* Submit */
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rp);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
  camera_on_input_event(&state.camera, input_event);

  /* Use the persistent keys_down array rather than checking event_type.
   * The main loop only dispatches one event per frame; if a MOUSE_MOVE fires
   * in the same glfwPollEvents() call as KEY_DOWN, MOUSE_MOVE overwrites the
   * event type and KEY_DOWN would never be seen.  keys_down[] is always a
   * reliable snapshot of which keys are currently held. */
  state.camera.keys.up    = input_event->keys_down[KEY_W];
  state.camera.keys.down  = input_event->keys_down[KEY_S];
  state.camera.keys.left  = input_event->keys_down[KEY_A];
  state.camera.keys.right = input_event->keys_down[KEY_D];
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layout_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layout_texture)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  wgpu_destroy_texture(&state.lightmap_tex);

  for (uint32_t t = 0; t < state.num_textures; t++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.textures[t].bind_group)
    wgpu_destroy_texture(&state.textures[t].gpu_tex);
  }

  /* Free PVS data */
  q2_cluster_faces_destroy(state.cluster_faces, state.num_pvs_clusters);
  state.cluster_faces = NULL;

  /* Free per-face mesh data */
  free(state.face_meshes);
  state.face_meshes = NULL;

  /* Free master vertex array */
  free(state.master_vertices);
  state.master_vertices = NULL;

  q2_bsp_destroy(&state.bsp);
  q2_pak_destroy(&state.pak);
}

/* -------------------------------------------------------------------------- *
 * Main Entry Point
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  wgpu_start(&(wgpu_desc_t){
    .title          = "Quake 2 BSP Renderer",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* q2_vertex_shader_wgsl = CODE(
  struct Uniforms {
    mvp : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) tex_uv : vec2f,
    @location(2) lm_uv : vec2f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) tex_uv : vec2f,
    @location(1) lm_uv : vec2f,
  };

  @vertex
  fn main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = uniforms.mvp * vec4f(in.position, 1.0);
    out.tex_uv = in.tex_uv;
    out.lm_uv = in.lm_uv;
    return out;
  }
);

static const char* q2_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var tex_sampler : sampler;
  @group(0) @binding(2) var lightmap_tex : texture_2d<f32>;
  @group(1) @binding(0) var diffuse_tex : texture_2d<f32>;

  @fragment
  fn main(
    @location(0) tex_uv : vec2f,
    @location(1) lm_uv : vec2f,
  ) -> @location(0) vec4f {
    let diffuse = textureSample(diffuse_tex, tex_sampler, tex_uv);
    let light = textureSample(lightmap_tex, tex_sampler, lm_uv);
    return vec4f(diffuse.rgb * light.rgb, diffuse.a);
  }
);
// clang-format on
