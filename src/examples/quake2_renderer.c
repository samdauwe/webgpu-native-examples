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

/* Brush content flags */
#define Q2_CONTENTS_SOLID 0x00000001
#define Q2_CONTENTS_WINDOW 0x00000002
#define Q2_CONTENTS_MONSTER 0x00000080
#define Q2_CONTENTS_DEADMONSTER 0x00000100

/* Maximum number of BSP maps discoverable from a PAK file */
#define Q2_MAX_MAPS 32

/* Gizmo axis length in Quake 2 world units (1 unit ≈ 1 inch, ~64 = 5 ft) */
#define Q2_GIZMO_SIZE 64.0f

/* Debug / visualization view modes */
typedef enum {
  Q2_VIEW_NORMAL    = 0, /* Textured + lightmapped (default) */
  Q2_VIEW_COLORED   = 1, /* Non-textured, random color per face */
  Q2_VIEW_COLLISION = 2, /* Brush collision volumes */
} q2_view_mode_t;

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

/* Map info entry discovered from PAK directory */
typedef struct {
  char pak_path[Q2_PAK_FILENAME_LEN]; /* e.g. "maps/base1.bsp" */
  char display_name[64];              /* e.g. "Outer Base" from entity string */
  char description[128];              /* Level type + short description */
} q2_map_info_t;

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

/**
 * @brief Extract the "message" field from a BSP worldspawn entity string.
 *
 * The worldspawn entity (classname "worldspawn") typically contains a
 * "message" key with the human-readable map name/title.
 *
 * @param entity_str  Null-terminated entity string from the BSP file.
 * @param out_message Output buffer for the extracted message.
 * @param max_len     Size of the output buffer.
 * @return true if a message was found.
 */
static bool q2_extract_map_message(const char* entity_str, char* out_message,
                                   uint32_t max_len)
{
  out_message[0] = '\0';
  if (!entity_str) {
    return false;
  }

  /* Find the worldspawn entity block (should be the first entity) */
  const char* ws = strstr(entity_str, "\"worldspawn\"");
  if (!ws) {
    return false;
  }

  /* Walk back to find the opening brace */
  const char* block_start = ws;
  while (block_start > entity_str && *block_start != '{') {
    block_start--;
  }

  /* Find the closing brace */
  const char* block_end = strchr(ws, '}');
  if (!block_end) {
    return false;
  }

  /* Look for "message" key within the worldspawn block */
  const char* msg_key = strstr(block_start, "\"message\"");
  if (!msg_key || msg_key >= block_end) {
    return false;
  }

  /* Skip past the key and find the value string */
  const char* val_start = strchr(msg_key + 9, '"');
  if (!val_start || val_start >= block_end) {
    return false;
  }
  val_start++; /* Skip opening quote */

  const char* val_end = strchr(val_start, '"');
  if (!val_end || val_end >= block_end) {
    return false;
  }

  uint32_t len = (uint32_t)(val_end - val_start);
  if (len >= max_len) {
    len = max_len - 1;
  }
  memcpy(out_message, val_start, len);
  out_message[len] = '\0';
  return len > 0;
}

/**
 * @brief Discover all BSP maps in the PAK archive and extract metadata.
 *
 * Scans the PAK directory for maps BSP entries. For each map found,
 * partially parses the BSP entity lump to extract the map title ("message"
 * from the worldspawn entity).
 *
 * @param pak       Loaded PAK archive.
 * @param maps      Output array of map info entries.
 * @param num_maps  Output: number of maps found.
 * @param max_maps  Maximum entries in the maps array.
 */
static void q2_discover_maps(const q2_pak_t* pak, q2_map_info_t* maps,
                             uint32_t* num_maps, uint32_t max_maps)
{
  *num_maps = 0;

  for (uint32_t i = 0; i < pak->num_entries && *num_maps < max_maps; i++) {
    const char* name = pak->entries[i].filename;
    uint32_t len     = (uint32_t)strnlen(name, Q2_PAK_FILENAME_LEN);

    /* Match maps BSP entries (maps/xxx.bsp) */
    if (len <= 9 || strncmp(name, "maps/", 5) != 0
        || strncmp(name + len - 4, ".bsp", 4) != 0) {
      continue;
    }

    q2_map_info_t* info = &maps[*num_maps];
    strncpy(info->pak_path, name, Q2_PAK_FILENAME_LEN - 1);
    info->pak_path[Q2_PAK_FILENAME_LEN - 1] = '\0';

    /* Extract just the map name (without path and extension) for display */
    const char* base  = name + 5;    /* skip "maps/" */
    uint32_t base_len = len - 5 - 4; /* without .bsp */
    char map_name[48] = {0};
    if (base_len >= sizeof(map_name)) {
      base_len = sizeof(map_name) - 1;
    }
    memcpy(map_name, base, base_len);

    /* Try to extract the "message" (display name) from BSP entity string */
    info->display_name[0] = '\0';
    info->description[0]  = '\0';

    uint32_t bsp_size       = 0;
    const uint8_t* bsp_data = q2_pak_get_data(pak, &pak->entries[i], &bsp_size);
    if (bsp_data && bsp_size >= sizeof(q2_bsp_header_t)) {
      const q2_bsp_header_t* hdr = (const q2_bsp_header_t*)bsp_data;
      if (hdr->magic == Q2_BSP_MAGIC && hdr->version == Q2_BSP_VERSION) {
        /* Read entity lump to extract message */
        const q2_lump_t* ent_lump = &hdr->lumps[Q2_LUMP_ENTITIES];
        if (ent_lump->offset + ent_lump->length <= bsp_size
            && ent_lump->length > 0) {
          /* Temporarily null-terminate the entity string */
          char* ent_str = (char*)malloc(ent_lump->length + 1);
          if (ent_str) {
            memcpy(ent_str, bsp_data + ent_lump->offset, ent_lump->length);
            ent_str[ent_lump->length] = '\0';

            q2_extract_map_message(ent_str, info->display_name,
                                   sizeof(info->display_name));
            free(ent_str);
          }
        }

        /* Build description with face/vertex counts */
        uint32_t num_faces = hdr->lumps[Q2_LUMP_FACES].length / 20;
        uint32_t num_verts = hdr->lumps[Q2_LUMP_VERTICES].length / 12;
        snprintf(info->description, sizeof(info->description),
                 "%s — %u faces, %u vertices",
                 info->display_name[0] ? info->display_name : map_name,
                 num_faces, num_verts);
      }
    }

    /* Fallback: use filename as display name */
    if (info->display_name[0] == '\0') {
      snprintf(info->display_name, sizeof(info->display_name), "%s", map_name);
    }

    printf("[PAK] Map %u: %-20s \"%s\"\n", *num_maps, info->pak_path,
           info->display_name);
    (*num_maps)++;
  }

  printf("[PAK] Discovered %u BSP maps\n", *num_maps);
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
 * ========================================================================== */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations)
 * -------------------------------------------------------------------------- */

static const char* q2_vertex_shader_wgsl;
static const char* q2_fragment_shader_wgsl;
static const char* q2_skybox_vs_wgsl;
static const char* q2_skybox_fs_wgsl;
static const char* q2_debug_vs_wgsl;
static const char* q2_debug_fs_wgsl;
static const char* q2_md2_vs_wgsl;
static const char* q2_md2_fs_wgsl;

/* -------------------------------------------------------------------------- *
 * Lightmap Atlas BSP Tree Allocator
 *
 * Allocates rectangular regions on a 512x512 lightmap atlas using a binary
 * space partitioning tree.
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
 * @brief Copy a face's lightmap data into the atlas with 4x brightness  scaling
 * and proportional clamping.
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
 * PCX Image Decoder (Quake 2 sky textures)
 *
 * Quake 2 uses PCX format for sky environment textures stored in the PAK
 * archive under env/{skyname}{suffix}.pcx. The format uses RLE compression
 * with a 256-color palette appended at the end of the file.
 * -------------------------------------------------------------------------- */

/**
 * @brief Decode a Quake 2 PCX image to RGBA. Handles RLE decompression and
 *        palette lookup. Caller must free() the returned buffer.
 */
static uint8_t* q2_pcx_decode(const uint8_t* data, uint32_t size,
                              uint32_t* out_w, uint32_t* out_h)
{
  if (size < 128 + 769) {
    return NULL; /* Too small for header + palette */
  }

  /* PCX header fields */
  uint8_t manufacturer = data[0];
  uint8_t version      = data[1];
  uint8_t encoding     = data[2];
  uint8_t bpp          = data[3];

  if (manufacturer != 0x0A || version != 5 || bpp != 8) {
    return NULL;
  }

  uint16_t xmin = (uint16_t)(data[4] | (data[5] << 8));
  uint16_t ymin = (uint16_t)(data[6] | (data[7] << 8));
  uint16_t xmax = (uint16_t)(data[8] | (data[9] << 8));
  uint16_t ymax = (uint16_t)(data[10] | (data[11] << 8));

  uint32_t w = (uint32_t)(xmax - xmin + 1);
  uint32_t h = (uint32_t)(ymax - ymin + 1);

  if (w == 0 || h == 0 || w > 4096 || h > 4096) {
    return NULL;
  }

  /* Read 256-color palette from last 769 bytes (0x0C marker + 768 RGB) */
  if (data[size - 769] != 0x0C) {
    return NULL;
  }
  const uint8_t* palette = data + size - 768;

  /* Decode palette indices (RLE or raw) */
  uint32_t total  = w * h;
  uint8_t* pixels = (uint8_t*)malloc(total);
  if (!pixels) {
    return NULL;
  }

  const uint8_t* src = data + 128; /* Pixel data starts after header */
  const uint8_t* end = data + size - 769;
  uint32_t idx       = 0;

  if (encoding) {
    /* RLE decoding */
    while (idx < total && src < end) {
      uint8_t byte = *src++;
      if ((byte & 0xC0) == 0xC0) {
        uint32_t run = byte & 0x3F;
        if (src >= end) {
          break;
        }
        byte = *src++;
        while (run-- > 0 && idx < total) {
          pixels[idx++] = byte;
        }
      }
      else {
        pixels[idx++] = byte;
      }
    }
  }
  else {
    /* Raw pixel data */
    uint32_t copy
      = total < (uint32_t)(end - src) ? total : (uint32_t)(end - src);
    memcpy(pixels, src, copy);
    idx = copy;
  }

  /* Convert palette indices to RGBA */
  uint8_t* rgba = (uint8_t*)malloc(w * h * 4);
  if (!rgba) {
    free(pixels);
    return NULL;
  }

  for (uint32_t i = 0; i < w * h; i++) {
    uint8_t pi      = pixels[i];
    rgba[i * 4 + 0] = palette[pi * 3 + 0];
    rgba[i * 4 + 1] = palette[pi * 3 + 1];
    rgba[i * 4 + 2] = palette[pi * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }

  free(pixels);
  *out_w = w;
  *out_h = h;
  return rgba;
}

/* -------------------------------------------------------------------------- *
 * Skybox Loading
 *
 * Extracts the "sky" key from the BSP worldspawn entity and loads 6 PCX face
 * textures from the PAK archive at env/{skyname}{suffix}.pcx.
 *
 * Face order matches the standard cubemap layout:
 *   0: +X (right), 1: -X (left), 2: +Y (up),
 *   3: -Y (down),  4: +Z (front), 5: -Z (back)
 *
 * Pixel transformations are applied per the q2-veldrid-viewer reference:
 *   - Side faces (rt, lf, ft, bk): horizontally flipped
 *   - Top/bottom faces (up, dn): XY transposed
 * -------------------------------------------------------------------------- */

/* Quake 2 sky texture suffixes and their cubemap face mapping */
static const char* q2_sky_suffixes[6] = {"rt", "lf", "up", "dn", "ft", "bk"};

/**
 * @brief Extract the sky texture name from the BSP entity string.
 *
 * Searches the worldspawn entity (first entity block) for the "sky" key
 * and returns the value (e.g., "unit1"). Returns false if not found.
 */
static bool q2_get_sky_name(const char* entities, char* out_name,
                            uint32_t name_size)
{
  if (!entities || !out_name || name_size == 0) {
    return false;
  }

  const char* sky_key = strstr(entities, "\"sky\"");
  if (!sky_key) {
    return false;
  }

  /* Skip past "sky" key and find the opening quote of the value */
  const char* q1 = strchr(sky_key + 5, '"');
  if (!q1) {
    return false;
  }

  const char* q2 = strchr(q1 + 1, '"');
  if (!q2) {
    return false;
  }

  uint32_t len = (uint32_t)(q2 - q1 - 1);
  if (len == 0 || len >= name_size) {
    return false;
  }

  memcpy(out_name, q1 + 1, len);
  out_name[len] = '\0';
  return true;
}

/**
 * @brief Load 6 sky face textures from the PAK and assemble a cubemap.
 *
 * @param pak      PAK archive.
 * @param sky_name Sky texture base name (e.g., "unit1").
 * @param out_data Output: interleaved RGBA pixel data for 6 faces.
 * @param out_size Output: cube face dimension (faces are square).
 * @return true on success.
 */
static bool q2_load_sky_faces(const q2_pak_t* pak, const char* sky_name,
                              uint8_t** out_data, uint32_t* out_size)
{
  uint8_t* face_pixels[6] = {NULL};
  uint32_t face_w[6]      = {0};
  uint32_t face_h[6]      = {0};
  uint32_t cube_size      = 0;

  for (int i = 0; i < 6; i++) {
    char path[128];
    snprintf(path, sizeof(path), "env/%s%s.pcx", sky_name, q2_sky_suffixes[i]);

    const q2_pak_entry_t* entry = q2_pak_find(pak, path);
    if (!entry) {
      printf("[Q2] Sky face missing: %s\n", path);
      goto fail;
    }

    uint32_t file_size       = 0;
    const uint8_t* file_data = q2_pak_get_data(pak, entry, &file_size);
    if (!file_data) {
      goto fail;
    }

    face_pixels[i]
      = q2_pcx_decode(file_data, file_size, &face_w[i], &face_h[i]);
    if (!face_pixels[i]) {
      printf("[Q2] Failed to decode: %s\n", path);
      goto fail;
    }

    /* Verify square and consistent size */
    if (face_w[i] != face_h[i]) {
      printf("[Q2] Sky face %s not square: %ux%u\n", path, face_w[i],
             face_h[i]);
      goto fail;
    }
    if (i == 0) {
      cube_size = face_w[0];
    }
    else if (face_w[i] != cube_size) {
      printf("[Q2] Sky face %s size mismatch: %u vs %u\n", path, face_w[i],
             cube_size);
      goto fail;
    }
  }

  /* Allocate interleaved cubemap data: 6 faces × size × size × 4 bytes */
  uint32_t face_bytes  = cube_size * cube_size * 4;
  uint32_t total_bytes = 6 * face_bytes;
  uint8_t* cubemap     = (uint8_t*)malloc(total_bytes);
  if (!cubemap) {
    goto fail;
  }

  /* Apply per-face pixel transformations and copy into cubemap layout */
  for (int f = 0; f < 6; f++) {
    uint8_t* dst       = cubemap + f * face_bytes;
    const uint8_t* src = face_pixels[f];

    for (uint32_t y = 0; y < cube_size; y++) {
      for (uint32_t x = 0; x < cube_size; x++) {
        uint32_t src_idx, dst_idx;

        if (f == 2 || f == 3) {
          /* Up / Down: XY transpose */
          src_idx = (y + x * cube_size) * 4;
          dst_idx = (x + y * cube_size) * 4;
        }
        else {
          /* Side faces (rt, lf, ft, bk): horizontal flip */
          src_idx = (x + y * cube_size) * 4;
          dst_idx = ((cube_size - 1 - x) + y * cube_size) * 4;
        }

        dst[dst_idx + 0] = src[src_idx + 0];
        dst[dst_idx + 1] = src[src_idx + 1];
        dst[dst_idx + 2] = src[src_idx + 2];
        dst[dst_idx + 3] = src[src_idx + 3];
      }
    }

    free(face_pixels[f]);
    face_pixels[f] = NULL;
  }

  *out_data = cubemap;
  *out_size = cube_size;
  return true;

fail:
  for (int i = 0; i < 6; i++) {
    free(face_pixels[i]);
  }
  return false;
}

/* -------------------------------------------------------------------------- *
 * Geometry Types and Helpers
 * -------------------------------------------------------------------------- */

/* Debug / gizmo / overlay vertex: position + RGB color (6 floats = 24 bytes).
 * Used by the gizmo line list, non-textured colored triangles, and brush
 * volume wireframes — all sharing the same simple debug shader pipeline. */
typedef struct {
  float pos[3];   /* World position (Q2 coordinate space) */
  float color[3]; /* Linear RGB [0..1] */
} q2_debug_vertex_t;

/* Render vertex: 7 floats = 28 bytes */
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

/* -------------------------------------------------------------------------- *
 * MD2 Model Format Structures & Constants
 *
 * MD2 is Quake 2's animated mesh format. Each file contains a static mesh
 * with multiple animation frames stored as compressed byte-triplet vertices.
 * Frame interpolation produces smooth animation at runtime.
 *
 * Ref: http://tfc.duke.free.fr/coding/md2-specs-en.html
 * -------------------------------------------------------------------------- */

#define Q2_MD2_MAGIC 0x32504449 /* "IDP2" in little-endian */
#define Q2_MD2_VERSION 8
#define Q2_MD2_MAX_SKINS 32
#define Q2_MD2_SKIN_NAME_LEN 64
#define Q2_MD2_FRAME_NAME_LEN 16
#define Q2_MD2_MAX_MODELS 256   /* Max unique model slots in a map */
#define Q2_MD2_MAX_ENTITIES 512 /* Max entity instances in a map */
#define Q2_MD2_ANIM_FPS 10.0f   /* Default MD2 animation speed */
#define Q2_MD2_CULL_DISTANCE                                                   \
  8192.0f /* Distance beyond which models are culled */

/* MD2 file header (68 bytes) */
typedef struct {
  int32_t magic;             /* "IDP2" = 0x32504449 */
  int32_t version;           /* Must be 8 */
  int32_t skin_width;        /* Texture width in pixels */
  int32_t skin_height;       /* Texture height in pixels */
  int32_t frame_size;        /* Bytes per frame */
  int32_t num_skins;         /* Number of skin names */
  int32_t num_vertices;      /* Number of vertices per frame */
  int32_t num_tex_coords;    /* Number of texture coordinates */
  int32_t num_triangles;     /* Number of triangles */
  int32_t num_gl_cmds;       /* Number of OpenGL commands */
  int32_t num_frames;        /* Number of animation frames */
  int32_t offset_skins;      /* Offset to skin names */
  int32_t offset_tex_coords; /* Offset to texture coordinates */
  int32_t offset_triangles;  /* Offset to triangle indices */
  int32_t offset_frames;     /* Offset to frame data */
  int32_t offset_gl_cmds;    /* Offset to GL commands */
  int32_t offset_end;        /* Offset to end of file */
} q2_md2_header_t;

/* MD2 compressed vertex (4 bytes) */
typedef struct {
  uint8_t v[3];         /* Compressed position [0-255] */
  uint8_t normal_index; /* Index into normal lookup table */
} q2_md2_vertex_t;

/* MD2 texture coordinate (4 bytes, pixel space) */
typedef struct {
  int16_t s, t;
} q2_md2_texcoord_t;

/* MD2 triangle (12 bytes) */
typedef struct {
  uint16_t vertex_indices[3];   /* Indices into vertex array */
  uint16_t texcoord_indices[3]; /* Indices into texcoord array */
} q2_md2_triangle_t;

/* MD2 frame header (40 bytes + variable vertex data) */
typedef struct {
  float scale[3];                   /* Scale for decompression */
  float translate[3];               /* Translation for decompression */
  char name[Q2_MD2_FRAME_NAME_LEN]; /* Frame name (animation group) */
  /* Followed by num_vertices q2_md2_vertex_t */
} q2_md2_frame_header_t;

/* MD2 model vertex for GPU rendering (32 bytes) */
typedef struct {
  float pos[3];      /* Current frame position (12 bytes) */
  float next_pos[3]; /* Next frame position for interpolation (12 bytes) */
  float tex_uv[2];   /* Texture coordinates (8 bytes) */
} q2_md2_render_vertex_t;

/* Parsed MD2 model (CPU-side data) */
typedef struct {
  /* Header info */
  int32_t num_vertices;
  int32_t num_triangles;
  int32_t num_frames;
  int32_t skin_width;
  int32_t skin_height;

  /* Decompressed frame vertices: [num_frames][num_vertices] */
  float* frame_verts; /* Flat array: frame * num_vertices * 3 */

  /* Triangle indices */
  q2_md2_triangle_t* triangles;

  /* Texture coordinates (normalized 0-1) */
  float* tex_coords; /* [num_tex_coords * 2] */
  int32_t num_tex_coords;

  /* Skin name from file (for texture lookup) */
  char skin_name[Q2_MD2_SKIN_NAME_LEN];
} q2_md2_model_t;

/* Entity instance referencing an MD2 model */
typedef struct {
  float origin[3];     /* Q2 world position (X right, Y forward, Z up) */
  float angle;         /* Yaw rotation in degrees */
  int32_t model_index; /* Index into md2_models[] array (-1 = invalid) */
  int32_t cluster;     /* BSP cluster for PVS culling (-1 = unknown) */

  /* Gameplay classification: true = object doesn't actively animate in the
   * world (barrels, items, weapons on ground). Their MD2 frames are for
   * event animations (explosion, pickup), NOT idle loops. */
  bool is_gameplay_static;

  /* Animation state */
  int32_t current_frame;
  int32_t next_frame;
  float interp; /* Interpolation factor [0,1] between frames */
} q2_md2_entity_t;

/* MD2 model resource (GPU texture + shared model data) */
typedef struct {
  q2_md2_model_t model;
  wgpu_texture_t gpu_tex;
  bool loaded;
} q2_md2_resource_t;

/* MD2 pre-calculated normal lookup table (162 unit vectors).
 * Each MD2 vertex stores a 1-byte index into this table. */
// clang-format off
static const float q2_md2_normals[162][3] = {
  {-0.525731f,  0.000000f,  0.850651f}, {-0.442863f,  0.238856f,  0.864188f},
  {-0.295242f,  0.000000f,  0.955423f}, {-0.309017f,  0.500000f,  0.809017f},
  {-0.162460f,  0.262866f,  0.951056f}, { 0.000000f,  0.000000f,  1.000000f},
  { 0.000000f,  0.850651f,  0.525731f}, {-0.147621f,  0.716567f,  0.681718f},
  { 0.147621f,  0.716567f,  0.681718f}, { 0.000000f,  0.525731f,  0.850651f},
  { 0.309017f,  0.500000f,  0.809017f}, { 0.525731f,  0.000000f,  0.850651f},
  { 0.295242f,  0.000000f,  0.955423f}, { 0.442863f,  0.238856f,  0.864188f},
  { 0.162460f,  0.262866f,  0.951056f}, {-0.681718f,  0.147621f,  0.716567f},
  {-0.809017f,  0.309017f,  0.500000f}, {-0.587785f,  0.425325f,  0.688191f},
  {-0.850651f,  0.525731f,  0.000000f}, {-0.864188f,  0.442863f,  0.238856f},
  {-0.716567f,  0.681718f,  0.147621f}, {-0.688191f,  0.587785f,  0.425325f},
  {-0.500000f,  0.809017f,  0.309017f}, {-0.238856f,  0.864188f,  0.442863f},
  {-0.425325f,  0.688191f,  0.587785f}, {-0.716567f,  0.681718f, -0.147621f},
  {-0.500000f,  0.809017f, -0.309017f}, {-0.525731f,  0.850651f,  0.000000f},
  { 0.000000f,  0.850651f, -0.525731f}, {-0.238856f,  0.864188f, -0.442863f},
  { 0.000000f,  0.955423f, -0.295242f}, {-0.262866f,  0.951056f, -0.162460f},
  { 0.000000f,  1.000000f,  0.000000f}, { 0.000000f,  0.955423f,  0.295242f},
  {-0.262866f,  0.951056f,  0.162460f}, { 0.238856f,  0.864188f,  0.442863f},
  { 0.262866f,  0.951056f,  0.162460f}, { 0.500000f,  0.809017f,  0.309017f},
  { 0.238856f,  0.864188f, -0.442863f}, { 0.262866f,  0.951056f, -0.162460f},
  { 0.500000f,  0.809017f, -0.309017f}, { 0.850651f,  0.525731f,  0.000000f},
  { 0.716567f,  0.681718f,  0.147621f}, { 0.716567f,  0.681718f, -0.147621f},
  { 0.525731f,  0.850651f,  0.000000f}, { 0.425325f,  0.688191f,  0.587785f},
  { 0.864188f,  0.442863f,  0.238856f}, { 0.688191f,  0.587785f,  0.425325f},
  { 0.809017f,  0.309017f,  0.500000f}, { 0.681718f,  0.147621f,  0.716567f},
  { 0.587785f,  0.425325f,  0.688191f}, { 0.955423f,  0.295242f,  0.000000f},
  { 1.000000f,  0.000000f,  0.000000f}, { 0.951056f,  0.162460f,  0.262866f},
  { 0.850651f, -0.525731f,  0.000000f}, { 0.955423f, -0.295242f,  0.000000f},
  { 0.864188f, -0.442863f,  0.238856f}, { 0.951056f, -0.162460f,  0.262866f},
  { 0.809017f, -0.309017f,  0.500000f}, { 0.681718f, -0.147621f,  0.716567f},
  { 0.850651f,  0.000000f,  0.525731f}, { 0.864188f,  0.442863f, -0.238856f},
  { 0.809017f,  0.309017f, -0.500000f}, { 0.951056f,  0.162460f, -0.262866f},
  { 0.525731f,  0.000000f, -0.850651f}, { 0.681718f,  0.147621f, -0.716567f},
  { 0.681718f, -0.147621f, -0.716567f}, { 0.850651f,  0.000000f, -0.525731f},
  { 0.809017f, -0.309017f, -0.500000f}, { 0.864188f, -0.442863f, -0.238856f},
  { 0.951056f, -0.162460f, -0.262866f}, { 0.147621f,  0.716567f, -0.681718f},
  { 0.309017f,  0.500000f, -0.809017f}, { 0.425325f,  0.688191f, -0.587785f},
  { 0.442863f,  0.238856f, -0.864188f}, { 0.587785f,  0.425325f, -0.688191f},
  { 0.688191f,  0.587785f, -0.425325f}, {-0.147621f,  0.716567f, -0.681718f},
  {-0.309017f,  0.500000f, -0.809017f}, { 0.000000f,  0.525731f, -0.850651f},
  {-0.525731f,  0.000000f, -0.850651f}, {-0.442863f,  0.238856f, -0.864188f},
  {-0.295242f,  0.000000f, -0.955423f}, {-0.162460f,  0.262866f, -0.951056f},
  { 0.000000f,  0.000000f, -1.000000f}, { 0.295242f,  0.000000f, -0.955423f},
  { 0.162460f,  0.262866f, -0.951056f}, {-0.442863f, -0.238856f, -0.864188f},
  {-0.309017f, -0.500000f, -0.809017f}, {-0.162460f, -0.262866f, -0.951056f},
  { 0.000000f, -0.850651f, -0.525731f}, {-0.147621f, -0.716567f, -0.681718f},
  { 0.147621f, -0.716567f, -0.681718f}, { 0.000000f, -0.525731f, -0.850651f},
  { 0.309017f, -0.500000f, -0.809017f}, { 0.442863f, -0.238856f, -0.864188f},
  { 0.162460f, -0.262866f, -0.951056f}, { 0.238856f, -0.864188f, -0.442863f},
  { 0.500000f, -0.809017f, -0.309017f}, { 0.425325f, -0.688191f, -0.587785f},
  { 0.716567f, -0.681718f, -0.147621f}, { 0.688191f, -0.587785f, -0.425325f},
  { 0.587785f, -0.425325f, -0.688191f}, { 0.000000f, -0.955423f, -0.295242f},
  { 0.000000f, -1.000000f,  0.000000f}, { 0.262866f, -0.951056f, -0.162460f},
  { 0.000000f, -0.850651f,  0.525731f}, { 0.000000f, -0.955423f,  0.295242f},
  { 0.238856f, -0.864188f,  0.442863f}, { 0.262866f, -0.951056f,  0.162460f},
  { 0.500000f, -0.809017f,  0.309017f}, { 0.716567f, -0.681718f,  0.147621f},
  { 0.525731f, -0.850651f,  0.000000f}, {-0.238856f, -0.864188f, -0.442863f},
  {-0.500000f, -0.809017f, -0.309017f}, {-0.262866f, -0.951056f, -0.162460f},
  {-0.850651f, -0.525731f,  0.000000f}, {-0.716567f, -0.681718f, -0.147621f},
  {-0.716567f, -0.681718f,  0.147621f}, {-0.525731f, -0.850651f,  0.000000f},
  {-0.500000f, -0.809017f,  0.309017f}, {-0.238856f, -0.864188f,  0.442863f},
  {-0.262866f, -0.951056f,  0.162460f}, {-0.864188f, -0.442863f,  0.238856f},
  {-0.809017f, -0.309017f,  0.500000f}, {-0.688191f, -0.587785f,  0.425325f},
  {-0.681718f, -0.147621f,  0.716567f}, {-0.442863f, -0.238856f,  0.864188f},
  {-0.587785f, -0.425325f,  0.688191f}, {-0.309017f, -0.500000f,  0.809017f},
  {-0.147621f, -0.716567f,  0.681718f}, {-0.425325f, -0.688191f,  0.587785f},
  {-0.162460f, -0.262866f,  0.951056f}, { 0.442863f, -0.238856f,  0.864188f},
  { 0.162460f, -0.262866f,  0.951056f}, { 0.309017f, -0.500000f,  0.809017f},
  { 0.147621f, -0.716567f,  0.681718f}, { 0.000000f, -0.525731f,  0.850651f},
  { 0.425325f, -0.688191f,  0.587785f}, { 0.587785f, -0.425325f,  0.688191f},
  { 0.688191f, -0.587785f,  0.425325f}, {-0.955423f,  0.295242f,  0.000000f},
  {-0.951056f,  0.162460f,  0.262866f}, {-1.000000f,  0.000000f,  0.000000f},
  {-0.850651f,  0.000000f,  0.525731f}, {-0.955423f, -0.295242f,  0.000000f},
  {-0.951056f, -0.162460f,  0.262866f}, {-0.864188f,  0.442863f, -0.238856f},
  {-0.951056f,  0.162460f, -0.262866f}, {-0.809017f,  0.309017f, -0.500000f},
  {-0.864188f, -0.442863f, -0.238856f}, {-0.951056f, -0.162460f, -0.262866f},
  {-0.809017f, -0.309017f, -0.500000f}, {-0.681718f,  0.147621f, -0.716567f},
  {-0.681718f, -0.147621f, -0.716567f}, {-0.850651f,  0.000000f, -0.525731f},
  {-0.688191f,  0.587785f, -0.425325f}, {-0.587785f,  0.425325f, -0.688191f},
  {-0.425325f,  0.688191f, -0.587785f}, {-0.425325f, -0.688191f, -0.587785f},
  {-0.587785f, -0.425325f, -0.688191f}, {-0.688191f, -0.587785f, -0.425325f},
};
// clang-format on

/* Classname-to-model-path mapping for Quake 2 entities.
 * Maps BSP entity classnames to their corresponding MD2 model paths within
 * the PAK archive. Only entities with known model paths are rendered.
 * skin_override: NULL = use MD2 header skin; non-NULL = specific skin path.
 * is_gameplay_static: true for objects that don't actively animate in the
 *   world (barrels sit still until shot, items sit on ground, etc.).
 *   Their MD2 frames are for event animations (explosion, pickup), not idle. */
typedef struct {
  const char* classname;
  const char* model_path;
  const char* skin_override;
  bool is_gameplay_static;
} q2_entity_model_map_t;

// clang-format off
static const q2_entity_model_map_t q2_entity_model_table[] = {
  /* Monsters — gameplay-animated (idle/walk/attack animations)
   * Soldier variants share the same MD2 but use different skins. */
  {"monster_infantry",       "models/monsters/infantry/tris.md2",  NULL, false},
  {"monster_soldier",        "models/monsters/soldier/tris.md2",   "models/monsters/soldier/skin.pcx", false},
  {"monster_soldier_light",  "models/monsters/soldier/tris.md2",   "models/monsters/soldier/skin_lt.pcx", false},
  {"monster_soldier_ss",     "models/monsters/soldier/tris.md2",   "models/monsters/soldier/skin_ss.pcx", false},
  {"monster_gunner",         "models/monsters/gunner/tris.md2",    NULL, false},
  {"monster_tank",           "models/monsters/tank/tris.md2",      "models/monsters/tank/skin.pcx", false},
  {"monster_tank_commander", "models/monsters/tank/tris.md2",      "models/monsters/tank/../ctank/skin.pcx", false},
  {"monster_chick",          "models/monsters/bitch/tris.md2",     NULL, false},
  {"monster_brain",          "models/monsters/brain/tris.md2",     NULL, false},
  {"monster_parasite",       "models/monsters/parasite/tris.md2",  NULL, false},
  {"monster_mutant",         "models/monsters/mutant/tris.md2",    NULL, false},
  {"monster_medic",          "models/monsters/medic/tris.md2",     NULL, false},
  {"monster_flipper",        "models/monsters/flipper/tris.md2",   NULL, false},
  {"monster_flyer",          "models/monsters/flyer/tris.md2",     NULL, false},
  {"monster_floater",        "models/monsters/float/tris.md2",     NULL, false},
  {"monster_hover",          "models/monsters/hover/tris.md2",     NULL, false},
  {"monster_gladiator",      "models/monsters/gladiatr/tris.md2",  NULL, false},
  {"monster_berserk",        "models/monsters/berserk/tris.md2",   NULL, false},
  {"monster_supertank",      "models/monsters/boss1/tris.md2",     NULL, false},
  {"monster_boss2",          "models/monsters/boss2/tris.md2",     NULL, false},
  {"monster_jorg",           "models/monsters/boss3/jorg/tris.md2", NULL, false},
  {"monster_makron",         "models/monsters/boss3/rider/tris.md2", NULL, false},
  {"monster_boss3_stand",    "models/monsters/boss3/rider/tris.md2", NULL, false},
  {"misc_insane",            "models/monsters/insane/tris.md2",    NULL, false},
  /* Misc objects — gameplay-animated */
  {"misc_banner",            "models/objects/banner/tris.md2",     NULL, false},
  /* Items / health / armor — gameplay-static (sit on ground until pickup) */
  {"item_health",            "models/items/healing/medium/tris.md2",   NULL, true},
  {"item_health_small",      "models/items/healing/stimpack/tris.md2", NULL, true},
  {"item_health_large",      "models/items/healing/large/tris.md2",    NULL, true},
  {"item_health_mega",       "models/items/mega_h/tris.md2",           NULL, true},
  {"item_armor_body",        "models/items/armor/body/tris.md2",       NULL, true},
  {"item_armor_combat",      "models/items/armor/combat/tris.md2",     NULL, true},
  {"item_armor_jacket",      "models/items/armor/jacket/tris.md2",     NULL, true},
  {"item_armor_shard",       "models/items/armor/shard/tris.md2",      NULL, true},
  /* Weapons: use g_ (ground/world) models, NOT v_ (viewmodel) — static */
  {"weapon_shotgun",         "models/weapons/g_shotg/tris.md2",   NULL, true},
  {"weapon_supershotgun",    "models/weapons/g_shotg2/tris.md2",  NULL, true},
  {"weapon_machinegun",      "models/weapons/g_machn/tris.md2",   NULL, true},
  {"weapon_chaingun",        "models/weapons/g_chain/tris.md2",   NULL, true},
  {"weapon_grenadelauncher", "models/weapons/g_launch/tris.md2",  NULL, true},
  {"weapon_rocketlauncher",  "models/weapons/g_rocket/tris.md2",  NULL, true},
  {"weapon_hyperblaster",    "models/weapons/g_hyperb/tris.md2",  NULL, true},
  {"weapon_railgun",         "models/weapons/g_rail/tris.md2",    NULL, true},
  {"weapon_bfg",             "models/weapons/g_bfg/tris.md2",     NULL, true},
  /* Ammo — static */
  {"ammo_shells",            "models/items/ammo/shells/medium/tris.md2",    NULL, true},
  {"ammo_bullets",           "models/items/ammo/bullets/medium/tris.md2",   NULL, true},
  {"ammo_cells",             "models/items/ammo/cells/medium/tris.md2",     NULL, true},
  {"ammo_rockets",           "models/items/ammo/rockets/medium/tris.md2",   NULL, true},
  {"ammo_grenades",          "models/items/ammo/grenades/medium/tris.md2",  NULL, true},
  {"ammo_slugs",             "models/items/ammo/slugs/medium/tris.md2",     NULL, true},
  /* Power-ups — static */
  {"item_quad",              "models/items/quaddama/tris.md2",    NULL, true},
  {"item_invulnerability",   "models/items/invulner/tris.md2",    NULL, true},
  {"item_power_screen",      "models/items/armor/screen/tris.md2", NULL, true},
  {"item_power_shield",      "models/items/armor/shield/tris.md2", NULL, true},
  {"item_breather",          "models/items/breather/tris.md2",    NULL, true},
  {"item_enviro",            "models/items/enviro/tris.md2",      NULL, true},
  {"item_silencer",          "models/items/silencer/tris.md2",    NULL, true},
  {"item_adrenaline",        "models/items/adrenal/tris.md2",     NULL, true},
  {"item_bandolier",         "models/items/band/tris.md2",        NULL, true},
  {"item_pack",              "models/items/pack/tris.md2",        NULL, true},
  {"item_ancient_head",      "models/items/c_head/tris.md2",      NULL, true},
  /* Keys — static */
  {"key_data_cd",            "models/items/keys/data_cd/tris.md2",  NULL, true},
  {"key_power_cube",         "models/items/keys/power/tris.md2",    NULL, true},
  {"key_pyramid",            "models/items/keys/pyramid/tris.md2",  NULL, true},
  {"key_data_spinner",       "models/items/keys/spinner/tris.md2",  NULL, true},
  {"key_pass",               "models/items/keys/pass/tris.md2",     NULL, true},
  {"key_blue_key",           "models/items/keys/key/tris.md2",      NULL, true},
  {"key_red_key",            "models/items/keys/red_key/tris.md2",  NULL, true},
  {"key_commander_head",     "models/monsters/commandr/head/tris.md2", NULL, true},
  /* Misc objects — gameplay-static (barrels sit still until shot) */
  {"misc_explobox",          "models/objects/barrels/tris.md2",   NULL, true},
  {"misc_satellite_dish",    "models/objects/satellite/tris.md2", NULL, true},
  {"misc_deadsoldier",       "models/deadbods/dude/tris.md2",     NULL, true},
};
// clang-format on

#define Q2_ENTITY_MODEL_TABLE_SIZE                                             \
  (sizeof(q2_entity_model_table) / sizeof(q2_entity_model_table[0]))

/**
 * @brief Compute texture-space UV for a vertex getTextureUV).
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
 * @brief Compute lightmap dimensions for a face getLightmapDimensions). Returns
 * width/height in luxels and the minimum texture-space UV coordinates.
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
 * @brief Find the best info_player_start entity in the BSP entity string.
 *
 * Quake 2 BSP files may contain several info_player_start entities:
 *  - Entities WITH a "targetname" field are scripted/triggered spawns used for
 *    chapter transitions or co-op.  They should NOT be used as the default
 *    single-player spawn point.
 *  - Entities WITHOUT a "targetname" are the plain, default spawn point that
 *    the single-player game uses when loading the map fresh.
 *
 * Strategy:
 *  1. Collect all info_player_start blocks.
 *  2. Return the first one that has NO "targetname".
 *  3. Fall back to the first one found if every block has a "targetname".
 *
 * Also handles both "angle" (single yaw value) and "angles" (pitch yaw roll)
 * entity fields for the spawn facing direction.
 *
 * @param entities  BSP entity string (null-terminated).
 * @param out_pos   Receives Q2 world coords (X right, Y fwd, Z up).
 * @param out_angle Receives the yaw angle in degrees.
 * @return true on success.
 */
static bool q2_find_player_start(const char* entities, float out_pos[3],
                                 float* out_angle)
{
  /* First candidate (any entity, kept as fallback) */
  bool found_fallback   = false;
  float fallback_pos[3] = {0};
  float fallback_angle  = 0.0f;

  const char* p = entities;
  while ((p = strstr(p, "info_player_start")) != NULL) {
    /* Walk back to find the containing opening brace. */
    const char* block_start = p;
    while (block_start > entities && *block_start != '{') {
      block_start--;
    }

    /* Find the closing brace of this entity block. */
    const char* block_end = strchr(p, '}');
    if (!block_end) {
      p++;
      continue;
    }

    /* --- Extract "origin" ------------------------------------------------ */
    float pos[3]    = {0};
    bool has_origin = false;

    const char* origin = strstr(block_start, "\"origin\"");
    if (origin && origin < block_end) {
      const char* q = strchr(origin + 8, '"');
      if (q && q < block_end) {
        q++;
        if (sscanf(q, "%f %f %f", &pos[0], &pos[1], &pos[2]) == 3) {
          has_origin = true;
        }
      }
    }

    if (!has_origin) {
      p++;
      continue;
    }

    /* --- Extract facing angle -------------------------------------------- */
    float angle = 0.0f;

    /* Try "angles" (3-vector: pitch yaw roll) first */
    const char* angles_key = strstr(block_start, "\"angles\"");
    if (angles_key && angles_key < block_end) {
      const char* q = strchr(angles_key + 8, '"');
      if (q && q < block_end) {
        float pitch = 0.0f, yaw = 0.0f, roll = 0.0f;
        if (sscanf(q + 1, "%f %f %f", &pitch, &yaw, &roll) >= 2) {
          angle = yaw;
        }
      }
    }
    else {
      /* Fall back to single-value "angle" (yaw only) */
      const char* angle_key = strstr(block_start, "\"angle\"");
      if (angle_key && angle_key < block_end) {
        const char* q = strchr(angle_key + 7, '"');
        if (q && q < block_end) {
          sscanf(q + 1, "%f", &angle);
        }
      }
    }

    /* --- Check whether this is a targeted (scripted) spawn --------------- */
    bool has_targetname
      = (strstr(block_start, "\"targetname\"") != NULL
         && strstr(block_start, "\"targetname\"") < block_end);

    if (!has_targetname) {
      /* Plain spawn — this is the default single-player start. Use it. */
      glm_vec3_copy((vec3){pos[0], pos[1], pos[2]}, out_pos);
      *out_angle = angle;
      return true;
    }

    /* Remember as fallback in case no plain spawn exists. */
    if (!found_fallback) {
      glm_vec3_copy((vec3){pos[0], pos[1], pos[2]}, fallback_pos);
      fallback_angle = angle;
      found_fallback = true;
    }

    p++;
  }

  /* All info_player_start entities had a targetname — use the fallback. */
  if (found_fallback) {
    glm_vec3_copy(fallback_pos, out_pos);
    *out_angle = fallback_angle;
    return true;
  }

  return false;
}

/* -------------------------------------------------------------------------- *
 * MD2 Model Loading
 *
 * Parses MD2 binary data from the PAK archive. Decompresses byte-encoded
 * vertices for all animation frames and normalizes texture coordinates.
 * -------------------------------------------------------------------------- */

/**
 * @brief Parse an MD2 model from raw binary data.
 *
 * @param data       Raw MD2 file data.
 * @param data_size  Size of the data in bytes.
 * @param model      Output model structure.
 * @return true on success.
 */
static bool q2_md2_parse(const uint8_t* data, uint32_t data_size,
                         q2_md2_model_t* model)
{
  memset(model, 0, sizeof(*model));

  if (data_size < sizeof(q2_md2_header_t)) {
    fprintf(stderr, "[MD2] File too small: %u bytes\n", data_size);
    return false;
  }

  const q2_md2_header_t* hdr = (const q2_md2_header_t*)data;

  if (hdr->magic != Q2_MD2_MAGIC || hdr->version != Q2_MD2_VERSION) {
    fprintf(stderr, "[MD2] Invalid header: magic=0x%08X version=%d\n",
            hdr->magic, hdr->version);
    return false;
  }

  if (hdr->num_vertices <= 0 || hdr->num_triangles <= 0
      || hdr->num_frames <= 0) {
    fprintf(stderr, "[MD2] Empty model: verts=%d tris=%d frames=%d\n",
            hdr->num_vertices, hdr->num_triangles, hdr->num_frames);
    return false;
  }

  model->num_vertices   = hdr->num_vertices;
  model->num_triangles  = hdr->num_triangles;
  model->num_frames     = hdr->num_frames;
  model->num_tex_coords = hdr->num_tex_coords;
  model->skin_width     = hdr->skin_width;
  model->skin_height    = hdr->skin_height;

  /* Extract first skin name if available */
  if (hdr->num_skins > 0 && hdr->offset_skins > 0
      && (uint32_t)(hdr->offset_skins + Q2_MD2_SKIN_NAME_LEN) <= data_size) {
    const char* skin_ptr = (const char*)(data + hdr->offset_skins);
    strncpy(model->skin_name, skin_ptr, Q2_MD2_SKIN_NAME_LEN - 1);
    model->skin_name[Q2_MD2_SKIN_NAME_LEN - 1] = '\0';
  }

  /* Parse triangles */
  if ((uint32_t)(hdr->offset_triangles
                 + hdr->num_triangles * (int32_t)sizeof(q2_md2_triangle_t))
      > data_size) {
    fprintf(stderr, "[MD2] Triangle data out of bounds\n");
    return false;
  }
  model->triangles = (q2_md2_triangle_t*)malloc((size_t)hdr->num_triangles
                                                * sizeof(q2_md2_triangle_t));
  memcpy(model->triangles, data + hdr->offset_triangles,
         (size_t)hdr->num_triangles * sizeof(q2_md2_triangle_t));

  /* Parse and normalize texture coordinates */
  if (hdr->num_tex_coords > 0 && hdr->offset_tex_coords > 0) {
    const q2_md2_texcoord_t* raw_tc
      = (const q2_md2_texcoord_t*)(data + hdr->offset_tex_coords);
    model->tex_coords
      = (float*)malloc((size_t)hdr->num_tex_coords * 2 * sizeof(float));
    float inv_w = (hdr->skin_width > 0) ? 1.0f / (float)hdr->skin_width : 1.0f;
    float inv_h
      = (hdr->skin_height > 0) ? 1.0f / (float)hdr->skin_height : 1.0f;
    for (int32_t i = 0; i < hdr->num_tex_coords; i++) {
      model->tex_coords[i * 2 + 0] = (float)raw_tc[i].s * inv_w;
      model->tex_coords[i * 2 + 1] = (float)raw_tc[i].t * inv_h;
    }
  }

  /* Decompress all frame vertices */
  model->frame_verts = (float*)malloc(
    (size_t)hdr->num_frames * (size_t)hdr->num_vertices * 3 * sizeof(float));

  for (int32_t f = 0; f < hdr->num_frames; f++) {
    uint32_t frame_offset
      = (uint32_t)(hdr->offset_frames + f * hdr->frame_size);
    if (frame_offset + sizeof(q2_md2_frame_header_t) > data_size) {
      fprintf(stderr, "[MD2] Frame %d header out of bounds\n", f);
      free(model->frame_verts);
      free(model->triangles);
      free(model->tex_coords);
      memset(model, 0, sizeof(*model));
      return false;
    }

    const q2_md2_frame_header_t* fhdr
      = (const q2_md2_frame_header_t*)(data + frame_offset);
    const q2_md2_vertex_t* verts
      = (const q2_md2_vertex_t*)(data + frame_offset
                                 + sizeof(q2_md2_frame_header_t));

    float* dst = &model->frame_verts[(size_t)f * (size_t)hdr->num_vertices * 3];
    for (int32_t v = 0; v < hdr->num_vertices; v++) {
      dst[v * 3 + 0]
        = fhdr->scale[0] * (float)verts[v].v[0] + fhdr->translate[0];
      dst[v * 3 + 1]
        = fhdr->scale[1] * (float)verts[v].v[1] + fhdr->translate[1];
      dst[v * 3 + 2]
        = fhdr->scale[2] * (float)verts[v].v[2] + fhdr->translate[2];
    }
  }

  return true;
}

/**
 * @brief Free MD2 model CPU-side resources.
 */
static void q2_md2_destroy(q2_md2_model_t* model)
{
  free(model->frame_verts);
  free(model->triangles);
  free(model->tex_coords);
  memset(model, 0, sizeof(*model));
}

/**
 * @brief Load an MD2 model's skin texture from the PAK file.
 *
 * Tries the skin path stored in the MD2 file, then falls back to looking
 * for a .pcx file next to the model.
 *
 * @param pak        PAK archive.
 * @param model_path MD2 model path in PAK (e.g.
 * "models/monsters/infantry/tris.md2").
 * @param model      Parsed MD2 model (for skin_name).
 * @param out_rgba   Output RGBA pixels (caller must free).
 * @param out_w      Output texture width.
 * @param out_h      Output texture height.
 * @return true if a texture was loaded.
 */
static bool q2_md2_load_skin(const q2_pak_t* pak, const char* model_path,
                             const q2_md2_model_t* model,
                             const char* skin_override, uint8_t** out_rgba,
                             uint32_t* out_w, uint32_t* out_h)
{
  *out_rgba = NULL;
  *out_w = *out_h = 0;

  /* If a skin override is specified, try it first (highest priority).
   * This allows entity types sharing an MD2 file to use different skins
   * (e.g., monster_soldier vs monster_soldier_light). */
  if (skin_override && skin_override[0] != '\0') {
    const q2_pak_entry_t* entry = q2_pak_find(pak, skin_override);
    if (entry) {
      uint32_t size       = 0;
      const uint8_t* data = q2_pak_get_data(pak, entry, &size);
      if (data) {
        *out_rgba = q2_pcx_decode(data, size, out_w, out_h);
        if (*out_rgba) {
          return true;
        }
      }
    }
    fprintf(stderr, "[MD2] WARNING: override skin not found: '%s'\n",
            skin_override);
  }

  /* Try skin name from MD2 header */
  if (model->skin_name[0] != '\0') {
    const q2_pak_entry_t* entry = q2_pak_find(pak, model->skin_name);
    if (entry) {
      uint32_t size       = 0;
      const uint8_t* data = q2_pak_get_data(pak, entry, &size);
      if (data) {
        *out_rgba = q2_pcx_decode(data, size, out_w, out_h);
        if (*out_rgba) {
          return true;
        }
      }
    }
  }

  /* Fallback: replace .md2 extension with .pcx in the model path */
  char skin_path[128];
  strncpy(skin_path, model_path, sizeof(skin_path) - 1);
  skin_path[sizeof(skin_path) - 1] = '\0';

  char* dot = strrchr(skin_path, '.');
  if (dot && (size_t)(dot - skin_path) < sizeof(skin_path) - 5) {
    strcpy(dot, ".pcx");
    const q2_pak_entry_t* entry = q2_pak_find(pak, skin_path);
    if (entry) {
      uint32_t size       = 0;
      const uint8_t* data = q2_pak_get_data(pak, entry, &size);
      if (data) {
        *out_rgba = q2_pcx_decode(data, size, out_w, out_h);
        if (*out_rgba) {
          return true;
        }
      }
    }
  }

  /* Fallback: try skin_i.pcx pattern (common for monsters) */
  {
    /* Get directory from model path */
    char dir[128];
    strncpy(dir, model_path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* last_slash     = strrchr(dir, '/');
    if (last_slash) {
      *(last_slash + 1) = '\0'; /* Keep trailing slash */
      char try_path[256];
      snprintf(try_path, sizeof(try_path), "%sskin.pcx", dir);
      const q2_pak_entry_t* entry = q2_pak_find(pak, try_path);
      if (entry) {
        uint32_t size       = 0;
        const uint8_t* data = q2_pak_get_data(pak, entry, &size);
        if (data) {
          *out_rgba = q2_pcx_decode(data, size, out_w, out_h);
          if (*out_rgba) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

/* -------------------------------------------------------------------------- *
 * BSP Entity Parsing for MD2 Models
 *
 * Scans the BSP entity string for entities that reference MD2 models.
 * Returns an array of entity instances with position, rotation, and model
 * index for rendering.
 * -------------------------------------------------------------------------- */

/**
 * @brief Look up the model path for a classname in the entity model table.
 * @return Model path string, or NULL if not found.
 */
static const q2_entity_model_map_t* q2_entity_lookup(const char* classname)
{
  for (uint32_t i = 0; i < Q2_ENTITY_MODEL_TABLE_SIZE; i++) {
    if (strcmp(classname, q2_entity_model_table[i].classname) == 0) {
      return &q2_entity_model_table[i];
    }
  }
  return NULL;
}

/**
 * @brief Find the BSP cluster containing a Q2 world position.
 *
 * Used during entity loading to pre-assign clusters for PVS culling.
 */
static int32_t q2_entity_get_cluster(const q2_bsp_map_t* map,
                                     const float pos[3])
{
  int32_t node_id = 0;
  while (node_id >= 0) {
    const q2_node_t* node   = &map->nodes[node_id];
    const q2_plane_t* plane = &map->planes[node->plane];
    float d;
    if (plane->type < 3) {
      d = pos[plane->type] - plane->dist;
    }
    else {
      d = pos[0] * plane->normal[0] + pos[1] * plane->normal[1]
          + pos[2] * plane->normal[2] - plane->dist;
    }
    node_id = (d >= 0) ? node->front_child : node->back_child;
  }
  int32_t leaf_idx = -(node_id + 1);
  if (leaf_idx >= 0 && (uint32_t)leaf_idx < map->num_leaves) {
    return (int32_t)map->leaves[leaf_idx].cluster;
  }
  return -1;
}

/**
 * @brief Parse BSP entity string and extract all renderable MD2 entities.
 *
 * For each entity block in the BSP entities lump, checks if the classname
 * maps to a known MD2 model. If so, extracts origin and angle and adds it
 * to the entity list.
 *
 * @param entities    BSP entity string.
 * @param map         BSP map (for cluster lookup).
 * @param out_entities Output array (caller must free).
 * @param out_count    Number of entities found.
 * @param model_paths  Output: unique model paths referenced (caller must
 * manage).
 * @param model_count  In/out: number of unique model paths.
 * @return true on success.
 */
static bool q2_parse_md2_entities(const char* entities, const q2_bsp_map_t* map,
                                  q2_md2_entity_t** out_entities,
                                  uint32_t* out_count, const char** model_paths,
                                  const char** skin_overrides,
                                  uint32_t* model_count)
{
  *out_entities = NULL;
  *out_count    = 0;

  if (!entities) {
    return false;
  }

  /* First pass: count entities */
  uint32_t capacity = 128;
  q2_md2_entity_t* ents
    = (q2_md2_entity_t*)malloc(capacity * sizeof(q2_md2_entity_t));
  uint32_t count = 0;

  const char* p = entities;
  while (*p) {
    /* Find next entity block */
    const char* brace_open = strchr(p, '{');
    if (!brace_open) {
      break;
    }
    const char* brace_close = strchr(brace_open, '}');
    if (!brace_close) {
      break;
    }

    /* Extract classname */
    char classname[64] = {0};
    const char* cn_key = strstr(brace_open, "\"classname\"");
    if (cn_key && cn_key < brace_close) {
      const char* q1 = strchr(cn_key + 11, '"');
      if (q1 && q1 < brace_close) {
        const char* q2 = strchr(q1 + 1, '"');
        if (q2 && q2 < brace_close) {
          uint32_t len = (uint32_t)(q2 - q1 - 1);
          if (len > 0 && len < sizeof(classname)) {
            memcpy(classname, q1 + 1, len);
            classname[len] = '\0';
          }
        }
      }
    }

    if (classname[0] == '\0') {
      p = brace_close + 1;
      continue;
    }

    /* Look up model entry (path + skin override + static flag) */
    const q2_entity_model_map_t* entry = q2_entity_lookup(classname);
    if (!entry) {
      p = brace_close + 1;
      continue;
    }

    const char* model_path    = entry->model_path;
    const char* skin_override = entry->skin_override;

    /* Find or add model slot to unique list.
     * Deduplication key is (model_path, skin_override) so that entity types
     * sharing the same MD2 file but needing different skins (e.g. soldier
     * variants) each get their own model resource slot with correct texture. */
    int32_t model_idx = -1;
    for (uint32_t i = 0; i < *model_count; i++) {
      if (strcmp(model_paths[i], model_path) == 0) {
        /* Same model path — also check skin override match */
        const char* existing_skin = skin_overrides[i];
        if (skin_override == NULL && existing_skin == NULL) {
          model_idx = (int32_t)i;
          break;
        }
        if (skin_override && existing_skin
            && strcmp(skin_override, existing_skin) == 0) {
          model_idx = (int32_t)i;
          break;
        }
      }
    }
    if (model_idx < 0 && *model_count < Q2_MD2_MAX_MODELS) {
      model_idx                    = (int32_t)*model_count;
      model_paths[*model_count]    = model_path;
      skin_overrides[*model_count] = skin_override;
      (*model_count)++;
    }
    if (model_idx < 0) {
      p = brace_close + 1;
      continue;
    }

    /* Extract origin */
    float origin[3]     = {0};
    const char* org_key = strstr(brace_open, "\"origin\"");
    if (org_key && org_key < brace_close) {
      const char* q1 = strchr(org_key + 8, '"');
      if (q1 && q1 < brace_close) {
        sscanf(q1 + 1, "%f %f %f", &origin[0], &origin[1], &origin[2]);
      }
    }

    /* Extract angle */
    float angle            = 0.0f;
    const char* angles_key = strstr(brace_open, "\"angles\"");
    if (angles_key && angles_key < brace_close) {
      const char* q1 = strchr(angles_key + 8, '"');
      if (q1 && q1 < brace_close) {
        float pitch = 0, yaw = 0, roll = 0;
        if (sscanf(q1 + 1, "%f %f %f", &pitch, &yaw, &roll) >= 2) {
          angle = yaw;
        }
      }
    }
    else {
      const char* angle_key = strstr(brace_open, "\"angle\"");
      if (angle_key && angle_key < brace_close) {
        const char* q1 = strchr(angle_key + 7, '"');
        if (q1 && q1 < brace_close) {
          sscanf(q1 + 1, "%f", &angle);
        }
      }
    }

    /* Grow array if needed */
    if (count >= capacity) {
      capacity *= 2;
      ents
        = (q2_md2_entity_t*)realloc(ents, capacity * sizeof(q2_md2_entity_t));
    }

    /* Store entity */
    ents[count].origin[0]          = origin[0];
    ents[count].origin[1]          = origin[1];
    ents[count].origin[2]          = origin[2];
    ents[count].angle              = angle;
    ents[count].model_index        = model_idx;
    ents[count].cluster            = q2_entity_get_cluster(map, origin);
    ents[count].is_gameplay_static = entry->is_gameplay_static;
    ents[count].current_frame      = 0;
    ents[count].next_frame = 0; /* Safe default; corrected after model load */
    ents[count].interp     = 0.0f;
    count++;

    p = brace_close + 1;
  }

  if (count == 0) {
    free(ents);
    return false;
  }

  *out_entities = ents;
  *out_count    = count;
  printf("[MD2] Found %u entities referencing %u unique models\n", count,
         *model_count);
  return true;
}

/* -------------------------------------------------------------------------- *
 * MD2 GPU Vertex Buffer Building
 *
 * Builds triangulated vertex data for a given frame pair (current + next)
 * with texture coordinates, ready for GPU upload. The vertex shader will
 * linearly interpolate between current and next frame positions.
 * -------------------------------------------------------------------------- */

/**
 * @brief Build GPU vertex buffer data for one MD2 model at a specific frame.
 *
 * @param model       Parsed MD2 model.
 * @param frame_cur   Current animation frame index.
 * @param frame_next  Next animation frame index (for interpolation).
 * @param out_verts   Output vertex array (caller must free).
 * @param out_count   Number of vertices generated.
 */
static void q2_md2_build_vertices(const q2_md2_model_t* model,
                                  int32_t frame_cur, int32_t frame_next,
                                  q2_md2_render_vertex_t** out_verts,
                                  uint32_t* out_count)
{
  uint32_t num_verts = (uint32_t)model->num_triangles * 3;
  q2_md2_render_vertex_t* verts
    = (q2_md2_render_vertex_t*)malloc(num_verts * sizeof(*verts));

  const float* cur_frame
    = &model->frame_verts[(size_t)frame_cur * (size_t)model->num_vertices * 3];
  const float* next_frame
    = &model->frame_verts[(size_t)frame_next * (size_t)model->num_vertices * 3];

  for (int32_t t = 0; t < model->num_triangles; t++) {
    const q2_md2_triangle_t* tri = &model->triangles[t];

    for (int k = 0; k < 3; k++) {
      q2_md2_render_vertex_t* rv = &verts[t * 3 + k];
      uint16_t vi                = tri->vertex_indices[k];
      uint16_t ti                = tri->texcoord_indices[k];

      /* Current frame position */
      rv->pos[0] = cur_frame[vi * 3 + 0];
      rv->pos[1] = cur_frame[vi * 3 + 1];
      rv->pos[2] = cur_frame[vi * 3 + 2];

      /* Next frame position */
      rv->next_pos[0] = next_frame[vi * 3 + 0];
      rv->next_pos[1] = next_frame[vi * 3 + 1];
      rv->next_pos[2] = next_frame[vi * 3 + 2];

      /* Texture coordinates */
      if (model->tex_coords && ti < model->num_tex_coords) {
        rv->tex_uv[0] = model->tex_coords[ti * 2 + 0];
        rv->tex_uv[1] = model->tex_coords[ti * 2 + 1];
      }
      else {
        rv->tex_uv[0] = 0.0f;
        rv->tex_uv[1] = 0.0f;
      }
    }
  }

  *out_verts = verts;
  *out_count = num_verts;
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
    /* Total byte count of the visibility lump (header + offsets + RLE data) */
    uint32_t lump_size = sizeof(uint32_t)
                         + num_clusters * (uint32_t)sizeof(q2_vis_offset_t)
                         + map->vis_data_size;

    while (c < num_clusters && v < lump_size) {
      uint8_t byte = lump_base[v];
      if (byte == 0) {
        /* RLE: skip clusters */
        v++;
        if (v >= lump_size) {
          break;
        }
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
 * Renderer Uniform Buffer Layout
 *
 * Packed struct mirroring the WGSL `Uniforms` block used by both the vertex
 * and fragment shaders.  Size must be a multiple of 16 bytes.
 *
 *  Offset  0 : mvp (mat4x4 = 64 bytes)
 *  Offset 64 : wireframe_mode (u32, 0 = normal, 1 = wireframe)
 *  Offset 68 : _pad[3]        (12 bytes padding to reach 80 bytes)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 mvp;                /* 64 bytes */
  uint32_t wireframe_mode; /* 4 bytes  */
  uint32_t _pad[3];        /* 12 bytes padding (total 80 = 5 × 16) */
} q2_uniforms_t;

/* MD2 model uniform buffer layout (96 bytes):
 *  Offset  0 : model_mvp (mat4x4 = 64 bytes)
 *  Offset 64 : interpolation (f32, 0.0 to 1.0 blend between frames)
 *  Offset 68 : _pad[7]       (28 bytes padding to reach 96 = 6 × 16)
 */
typedef struct {
  mat4 model_mvp;      /* 64 bytes */
  float interpolation; /* 4 bytes  */
  uint32_t _pad[7];    /* 28 bytes padding (total 96 = 6 × 16) */
} q2_md2_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Renderer State
 * -------------------------------------------------------------------------- */

static struct {
  /* Source data */
  q2_pak_t pak;
  q2_bsp_map_t bsp;

  /* Map list discovered from PAK */
  q2_map_info_t maps[Q2_MAX_MAPS];
  uint32_t num_maps;
  int32_t current_map_index;   /* Currently loaded map index (-1 = none) */
  int32_t requested_map_index; /* GUI-requested map switch (-1 = none) */

  /* Pending PAK file reload (from drag & drop) */
  struct {
    bool pending;
    char path[MAX_DROP_PATH_LEN];
  } pak_reload;

  /* Camera */
  camera_t camera;

  /* Timing */
  uint64_t last_frame_time;

  /* Performance stats (updated per-frame) */
  float fps;
  float frame_time_ms;
  uint32_t fps_frame_count;
  uint64_t fps_accum_start;

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

  /* Render settings */
  bool show_wireframe; /* Overlay wireframe edges on all geometry */

  /* Debug / visualization overlay */
  struct {
    q2_view_mode_t view_mode; /* Active geometry view mode */
    bool show_gizmo;          /* Render origin axes gizmo */

    /* Shared simple pipeline for colored debug geometry */
    WGPUBindGroupLayout bgl;
    WGPUBindGroup bg;
    WGPURenderPipeline line_pipeline; /* PrimitiveTopology_LineList */
    WGPURenderPipeline tri_pipeline;  /* PrimitiveTopology_TriangleList */
    WGPUPipelineLayout pipeline_layout;

    /* Gizmo: 3 axes (6 line-list vertices) at world origin */
    wgpu_buffer_t gizmo_vb;

    /* Non-textured colored geometry (random color per face, TriangleList) */
    wgpu_buffer_t colored_vb;
    uint32_t colored_vert_count;

    /* Brush collision volumes: AABB wireframe edges (LineList) */
    wgpu_buffer_t brush_vb;
    uint32_t brush_vert_count;
  } debug;

  /* Skybox */
  struct {
    bool enabled;
    WGPUTexture cubemap_handle;
    WGPUTextureView cubemap_view;
    WGPUSampler cubemap_sampler;
    wgpu_buffer_t uniform_buffer; /* view-rotation-projection inverse */
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
  } skybox;

  /* MD2 model rendering */
  struct {
    bool enabled;
    bool show_models; /* Toggle in GUI */

    /* Model resources (shared across instances) */
    q2_md2_resource_t models[Q2_MD2_MAX_MODELS];
    uint32_t num_models;

    /* Entity instances */
    q2_md2_entity_t* entities;
    uint32_t num_entities;

    /* GPU resources */
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t uniform_buffer; /* Dynamic uniform buffer for all draws */
    uint32_t vb_capacity;         /* Current VB capacity in bytes */
    uint32_t ub_capacity;         /* Current UB capacity in bytes */
    uint32_t uniform_alignment;   /* minUniformBufferOffsetAlignment (256) */
    WGPUBindGroupLayout bg_layout_shared;
    WGPUBindGroupLayout bg_layout_texture;
    WGPUBindGroup bg_shared;
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
    WGPUSampler sampler;

    /* Per-model texture bind groups (indexed by model_index) */
    WGPUBindGroup tex_bind_groups[Q2_MD2_MAX_MODELS];

    /* Pre-computed draw commands (populated before render pass) */
    struct {
      uint32_t vb_offset;    /* byte offset into vertex buffer */
      uint32_t vert_count;   /* number of vertices to draw */
      uint32_t ub_offset;    /* byte offset into uniform buffer (aligned) */
      int32_t model_index;   /* for texture bind group selection */
      uint32_t entity_index; /* index into entities[] for matrix data */
    } draw_cmds[Q2_MD2_MAX_ENTITIES];
    uint32_t num_draw_cmds;

    /* CPU staging buffers for batch upload */
    uint8_t* vb_staging;
    uint32_t vb_staging_cap;
    uint8_t* ub_staging;
    uint32_t ub_staging_cap;

    /* Visible entity indices (rebuilt on PVS change) */
    uint32_t* visible_entities;
    uint32_t num_visible;

    /* Timing */
    uint64_t anim_start_time;
  } md2;

  /* Render pass */
  WGPURenderPassColorAttachment color_att;
  WGPURenderPassDepthStencilAttachment depth_att;
  WGPURenderPassDescriptor rp_desc;

  WGPUBool initialized;
} state = {
  .current_map_index   = -1,
  .requested_map_index = -1,
  .current_leaf = -1,
  .prev_leaf    = -2, /* Force initial rebuild */
  .debug = {
    .view_mode  = Q2_VIEW_NORMAL,
    .show_gizmo = false,
  },
  .md2 = {
    .enabled     = false,
    .show_models = true,
  },
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

/**
 * @brief Load the PAK archive and discover available maps.
 *
 * Loads the entire PAK file into memory (serves as the cache for all map
 * switches) and scans the directory for BSP files with metadata.
 *
 * @return true on success.
 */
static bool load_pak_and_discover_maps(const char* override_path)
{
  const char* pak_path
    = override_path ? override_path :
      getenv("Q2_PAK_PATH") ?
                      getenv("Q2_PAK_PATH") :
                      "/home/sdauwe/GitHub/quake2generic/build/baseq2/pak0.pak";

  if (!q2_pak_load(pak_path, &state.pak)) {
    fprintf(stderr, "[ERROR] Could not load PAK: %s\n", pak_path);
    return false;
  }

  /* Discover all BSP maps in the PAK */
  q2_discover_maps(&state.pak, state.maps, &state.num_maps, Q2_MAX_MAPS);

  if (state.num_maps == 0) {
    fprintf(stderr, "[ERROR] No BSP maps found in PAK\n");
    return false;
  }

  return true;
}

/**
 * @brief Find the default map index to load initially.
 *
 * Prefers base1.bsp (first single-player level), then falls back to the
 * first map in the list.
 *
 * @return Index into state.maps[].
 */
static int32_t find_default_map_index(void)
{
  /* Preference order: base1 → demo1 → first available */
  static const char* preferred[] = {
    "maps/base1.bsp",
    "maps/demo1.bsp",
  };
  for (uint32_t p = 0; p < sizeof(preferred) / sizeof(preferred[0]); p++) {
    for (uint32_t i = 0; i < state.num_maps; i++) {
      if (strncmp(state.maps[i].pak_path, preferred[p], Q2_PAK_FILENAME_LEN)
          == 0) {
        return (int32_t)i;
      }
    }
  }
  return 0; /* Fallback: first discovered map */
}

/**
 * @brief Check if a file path has a .pak extension (case-insensitive).
 */
static bool path_is_pak(const char* path)
{
  if (!path) {
    return false;
  }
  size_t len = strlen(path);
  if (len < 4) {
    return false;
  }
  const char* ext = path + len - 4;
  return (ext[0] == '.' && (ext[1] == 'p' || ext[1] == 'P')
          && (ext[2] == 'a' || ext[2] == 'A')
          && (ext[3] == 'k' || ext[3] == 'K'));
}

/* Forward declarations for reload_pak */
static void cleanup_map_resources(wgpu_context_t* wgpu_context);
static bool switch_map(wgpu_context_t* wgpu_context, int32_t map_index);

/**
 * @brief Reload the entire PAK archive from a new file path.
 *
 * Performs a full teardown of the current map and PAK data, then loads the new
 * PAK, discovers maps, and switches to the default map.  Shared GPU resources
 * (pipelines, layouts, sampler, uniform buffer) are preserved since they are
 * PAK-independent.
 *
 * @param wgpu_context  WebGPU context.
 * @param pak_path      Absolute path to the new .pak file.
 * @return true on success; on failure the renderer is left with no loaded map.
 */
static bool reload_pak(wgpu_context_t* wgpu_context, const char* pak_path)
{
  printf("\n[Q2] === Reloading PAK: %s ===\n", pak_path);

  /* 1. Tear down current map resources (textures, geometry, MD2, skybox) */
  if (state.current_map_index >= 0) {
    cleanup_map_resources(wgpu_context);
    state.current_map_index   = -1;
    state.requested_map_index = -1;
  }

  /* 2. Destroy old PAK archive and reset map list */
  q2_pak_destroy(&state.pak);
  memset(state.maps, 0, sizeof(state.maps));
  state.num_maps = 0;

  /* 3. Load new PAK and discover maps */
  if (!load_pak_and_discover_maps(pak_path)) {
    fprintf(stderr, "[Q2] PAK reload failed: %s\n", pak_path);
    return false;
  }

  /* 4. Switch to the default map in the new PAK */
  int32_t default_map = find_default_map_index();
  if (!switch_map(wgpu_context, default_map)) {
    fprintf(stderr, "[Q2] Failed to load default map from new PAK\n");
    return false;
  }

  printf("[Q2] PAK reload complete: %u maps available\n", state.num_maps);
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

/* -------------------------------------------------------------------------- *
 * Debug / Visualization Helpers
 *
 * Utility functions for building debug geometry (gizmo axes, random-colored
 * faces, brush AABB wireframes) and the shared ColoredVertex pipeline.
 * -------------------------------------------------------------------------- */

/**
 * @brief Hash an integer seed into a bright, distinct RGB color.
 *
 * Uses a simple multiplicative hash to spread values across hue space.
 * Ensures minimum brightness of 0.35 so no color is too dark to see.
 */
static void debug_hash_color(uint32_t seed, float* r, float* g, float* b)
{
  uint32_t v = seed * 2654435761u ^ (seed >> 16);
  v          = (v ^ (v >> 17)) * 0x45d9f3b7u;
  float hue  = (float)(v & 0xFFFFu) / 65536.0f; /* 0..1 */
  float sat  = 0.7f;
  float val  = 0.9f;
  /* HSV to RGB (whole-domain) */
  float h6   = hue * 6.0f;
  int sector = (int)h6;
  float frac = h6 - (float)sector;
  float p    = val * (1.0f - sat);
  float q_   = val * (1.0f - sat * frac);
  float t_   = val * (1.0f - sat * (1.0f - frac));
  switch (sector % 6) {
    case 0:
      *r = val;
      *g = t_;
      *b = p;
      break;
    case 1:
      *r = q_;
      *g = val;
      *b = p;
      break;
    case 2:
      *r = p;
      *g = val;
      *b = t_;
      break;
    case 3:
      *r = p;
      *g = q_;
      *b = val;
      break;
    case 4:
      *r = t_;
      *g = p;
      *b = val;
      break;
    default:
      *r = val;
      *g = p;
      *b = q_;
      break;
  }
}

/**
 * @brief Append 24 LineList vertices (12 edges) forming a wireframe AABB.
 *
 * @param verts  Output array; must have room for at least *count+24 entries.
 * @param count  In/out: incremented by 24 after the call.
 * @param r,g,b  Edge color in linear RGB.
 */
static void debug_add_aabb_edges(q2_debug_vertex_t* verts, uint32_t* count,
                                 float xmin, float ymin, float zmin, float xmax,
                                 float ymax, float zmax, float r, float g,
                                 float b)
{
  /* 8 corners of the box */
  float cx[8] = {xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin};
  float cy[8] = {ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax};
  float cz[8] = {zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax};

  /* 12 edges as index pairs */
  static const uint8_t edges[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, /* bottom face */
    {4, 5}, {5, 6}, {6, 7}, {7, 4}, /* top face    */
    {0, 4}, {1, 5}, {2, 6}, {3, 7}, /* vertical    */
  };

  uint32_t base = *count;
  for (int e = 0; e < 12; e++) {
    uint8_t a                        = edges[e][0];
    uint8_t b_idx                    = edges[e][1];
    verts[base + e * 2].pos[0]       = cx[a];
    verts[base + e * 2].pos[1]       = cy[a];
    verts[base + e * 2].pos[2]       = cz[a];
    verts[base + e * 2].color[0]     = r;
    verts[base + e * 2].color[1]     = g;
    verts[base + e * 2].color[2]     = b;
    verts[base + e * 2 + 1].pos[0]   = cx[b_idx];
    verts[base + e * 2 + 1].pos[1]   = cy[b_idx];
    verts[base + e * 2 + 1].pos[2]   = cz[b_idx];
    verts[base + e * 2 + 1].color[0] = r;
    verts[base + e * 2 + 1].color[1] = g;
    verts[base + e * 2 + 1].color[2] = b;
  }
  *count += 24;
}

/* -------------------------------------------------------------------------- *
 * Debug Pipeline Initialization
 *
 * Creates a shared pipeline for position+color debug geometry.  Two topology
 * variants share one pipeline layout and one bind group (MVP matrix).
 * -------------------------------------------------------------------------- */

static void init_debug_pipelines(wgpu_context_t* wgpu_context)
{
  /* Bind group layout: one uniform buffer (MVP mat4) */
  WGPUBindGroupLayoutEntry bgl_entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Vertex,
    .buffer = (WGPUBufferBindingLayout){
      .type           = WGPUBufferBindingType_Uniform,
      .minBindingSize = sizeof(mat4), /* Only the first 64 bytes of uniforms */
    },
  };
  state.debug.bgl = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Q2 debug BGL"),
                            .entryCount = 1,
                            .entries    = &bgl_entry,
                          });
  ASSERT(state.debug.bgl != NULL);

  /* Bind group: share the main uniform buffer MVP (offset 0, size 64) */
  WGPUBindGroupEntry bg_entry = {
    .binding = 0,
    .buffer  = state.uniform_buffer.buffer,
    .offset  = 0,
    .size    = sizeof(mat4),
  };
  state.debug.bg = wgpuDeviceCreateBindGroup(wgpu_context->device,
                                             &(WGPUBindGroupDescriptor){
                                               .label  = STRVIEW("Q2 debug BG"),
                                               .layout = state.debug.bgl,
                                               .entryCount = 1,
                                               .entries    = &bg_entry,
                                             });
  ASSERT(state.debug.bg != NULL);

  /* Pipeline layout */
  state.debug.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Q2 debug pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.debug.bgl,
                          });
  ASSERT(state.debug.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule vs
    = wgpu_create_shader_module(wgpu_context->device, q2_debug_vs_wgsl);
  WGPUShaderModule fs
    = wgpu_create_shader_module(wgpu_context->device, q2_debug_fs_wgsl);

  /* Vertex layout: pos(vec3) + color(vec3) = 24 bytes */
  WGPU_VERTEX_BUFFER_LAYOUT(
    q2_debug, sizeof(q2_debug_vertex_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       offsetof(q2_debug_vertex_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(q2_debug_vertex_t, color)))

  WGPUDepthStencilState depth_stencil
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil.depthCompare = WGPUCompareFunction_LessEqual;

  /* Fragment target */
  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = NULL, /* opaque */
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Base descriptor shared between line and tri pipelines */
  WGPURenderPipelineDescriptor base_desc = {
    .layout = state.debug.pipeline_layout,
    .vertex = {
      .module      = vs,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &q2_debug_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = fs,
      .entryPoint  = STRVIEW("main"),
      .targetCount = 1,
      .targets     = &color_target,
    },
    .depthStencil = &depth_stencil,
    .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
  };

  /* LineList pipeline — gizmo and brush wireframes */
  WGPURenderPipelineDescriptor line_desc = base_desc;
  line_desc.label                        = STRVIEW("Q2 debug line pipeline");
  line_desc.primitive                    = (WGPUPrimitiveState){
                       .topology = WGPUPrimitiveTopology_LineList,
                       .cullMode = WGPUCullMode_None,
  };
  state.debug.line_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &line_desc);
  ASSERT(state.debug.line_pipeline != NULL);

  /* TriangleList pipeline — colored (non-textured) faces, no backface cull */
  WGPURenderPipelineDescriptor tri_desc = base_desc;
  tri_desc.label                        = STRVIEW("Q2 debug tri pipeline");
  tri_desc.primitive                    = (WGPUPrimitiveState){
                       .topology = WGPUPrimitiveTopology_TriangleList,
                       .cullMode = WGPUCullMode_None, /* No cull — avoids any facing issue */
  };
  state.debug.tri_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &tri_desc);
  ASSERT(state.debug.tri_pipeline != NULL);

  wgpuShaderModuleRelease(vs);
  wgpuShaderModuleRelease(fs);
}

/* -------------------------------------------------------------------------- *
 * Gizmo Vertex Buffer
 *
 * Three colored axes at the world origin: X=red, Y=green, Z=blue.
 * Rendered as a LineList (6 vertices, 3 lines).
 * -------------------------------------------------------------------------- */

static void init_gizmo(wgpu_context_t* wgpu_context)
{
  const float s = Q2_GIZMO_SIZE;
  /* Q2 coordinate axes (same space as BSP geometry): X=right, Z=up, Y=forward
   */
  const q2_debug_vertex_t gizmo_verts[6] = {
    {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, /* X origin */
    {{s, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},    /* X tip    — red   */
    {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, /* Y origin */
    {{0.0f, s, 0.0f}, {0.0f, 1.0f, 0.0f}},    /* Y tip    — green */
    {{0.0f, 0.0f, 0.0f}, {0.1f, 0.4f, 1.0f}}, /* Z origin */
    {{0.0f, 0.0f, s}, {0.1f, 0.4f, 1.0f}},    /* Z tip    — blue  */
  };
  state.debug.gizmo_vb = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 gizmo VB",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(gizmo_verts),
                    .initial.data = gizmo_verts,
                  });
}

/* -------------------------------------------------------------------------- *
 * Non-Textured Colored Geometry
 *
 * Builds a debug vertex buffer (pos+color) where every face's triangles share
 * a single deterministic random color derived from the face texture-info index.
 * Only renderable (non-sky, non-nodraw) faces are included.
 * -------------------------------------------------------------------------- */

static void init_debug_colored(wgpu_context_t* wgpu_context)
{
  /* Count total visible vertices across all faces */
  uint32_t total = 0;
  for (uint32_t fi = 0; fi < state.bsp.num_faces; fi++) {
    if (state.face_meshes[fi].batch_idx >= 0) {
      total += state.face_meshes[fi].vert_count;
    }
  }
  if (total == 0) {
    return;
  }

  q2_debug_vertex_t* verts
    = (q2_debug_vertex_t*)malloc(total * sizeof(q2_debug_vertex_t));
  if (!verts) {
    return;
  }

  uint32_t write = 0;
  for (uint32_t fi = 0; fi < state.bsp.num_faces; fi++) {
    const q2_face_mesh_t* fm = &state.face_meshes[fi];
    if (fm->batch_idx < 0) {
      continue;
    }

    /* One deterministic color per face (seed = texinfo index for consistency)
     */
    float r, g, b;
    uint16_t texinfo_idx = state.bsp.faces[fi].texinfo;
    debug_hash_color((uint32_t)texinfo_idx ^ (fi * 2654435761u >> 8), &r, &g,
                     &b);

    for (uint32_t vi = 0; vi < fm->vert_count; vi++) {
      const q2_render_vertex_t* src
        = &state.master_vertices[fm->start_vert + vi];
      verts[write].pos[0]   = src->pos[0];
      verts[write].pos[1]   = src->pos[1];
      verts[write].pos[2]   = src->pos[2];
      verts[write].color[0] = r;
      verts[write].color[1] = g;
      verts[write].color[2] = b;
      write++;
    }
  }

  state.debug.colored_vb = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 colored VB",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = write * sizeof(q2_debug_vertex_t),
                    .initial.data = verts,
                  });
  state.debug.colored_vert_count = write;
  free(verts);
}

/* -------------------------------------------------------------------------- *
 * Brush Collision Volume Wireframes
 *
 * Renders bounding boxes of BSP sub-models (models[1..N]):
 *   - Green  = has visible faces (door, platform, etc.)
 *   - Red    = no visible faces (trigger, clip brush)
 *
 * 12 edges × 2 vertices = 24 LineList vertices per model box.
 * -------------------------------------------------------------------------- */

static void init_debug_brush_volumes(wgpu_context_t* wgpu_context)
{
  if (state.bsp.num_models <= 1) {
    return; /* Only world model, no sub-models */
  }

  /* Maximum verts: 24 per sub-model (models[1..N]) */
  uint32_t max_verts = (state.bsp.num_models - 1) * 24;
  q2_debug_vertex_t* verts
    = (q2_debug_vertex_t*)malloc(max_verts * sizeof(q2_debug_vertex_t));
  if (!verts) {
    return;
  }

  uint32_t count = 0;
  for (uint32_t m = 1; m < state.bsp.num_models; m++) {
    const q2_model_t* mdl = &state.bsp.models[m];

    /* Color: green if the sub-model has drawable faces (solid prop/door/etc.)
     *        red for invisible volumes (triggers, clip brushes) */
    bool has_faces = false;
    for (int32_t fi = mdl->first_face; fi < mdl->first_face + mdl->num_faces
                                       && (uint32_t)fi < state.bsp.num_faces;
         fi++) {
      if (state.face_meshes[fi].batch_idx >= 0) {
        has_faces = true;
        break;
      }
    }
    float r = has_faces ? 0.1f : 1.0f;
    float g = has_faces ? 1.0f : 0.15f;
    float b = 0.1f;

    /* Convert Q2 AABB (X right, Y fwd, Z up) to WebGPU (X right, Y up, -Z fwd).
     * The Y-negate flips min/max: Q2_y_min → -Q2_y_min (WebGPU Z max),
     * Q2_y_max → -Q2_y_max (WebGPU Z min). */
    float wb_xmin = mdl->mins[0];
    float wb_ymin = mdl->mins[2];  /* Q2 Z min → WebGPU Y min */
    float wb_zmin = -mdl->maxs[1]; /* -Q2 Y max → WebGPU Z min */
    float wb_xmax = mdl->maxs[0];
    float wb_ymax = mdl->maxs[2];  /* Q2 Z max → WebGPU Y max */
    float wb_zmax = -mdl->mins[1]; /* -Q2 Y min → WebGPU Z max */

    debug_add_aabb_edges(verts, &count, wb_xmin, wb_ymin, wb_zmin, wb_xmax,
                         wb_ymax, wb_zmax, r, g, b);
  }

  if (count == 0) {
    free(verts);
    return;
  }

  state.debug.brush_vb = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 brush volume VB",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = count * sizeof(q2_debug_vertex_t),
                    .initial.data = verts,
                  });
  state.debug.brush_vert_count = count;
  free(verts);
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
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(q2_uniforms_t),
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
      .size    = sizeof(q2_uniforms_t),
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
        .frontFace = WGPUFrontFace_CW,
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
 * Skybox Initialization
 *
 * Loads sky textures from the PAK file and creates a separate render pipeline
 * for drawing the skybox cubemap. Uses a fullscreen triangle approach with
 * the inverse view-rotation-projection matrix to compute cubemap directions.
 * -------------------------------------------------------------------------- */

static void init_skybox(wgpu_context_t* wgpu_context)
{
  /* Extract sky name from BSP entity string */
  char sky_name[64] = {0};
  if (!state.bsp.entities
      || !q2_get_sky_name(state.bsp.entities, sky_name, sizeof(sky_name))) {
    printf("[Q2] No sky name found in BSP entities — skybox disabled\n");
    return;
  }

  printf("[Q2] Sky texture name: \"%s\"\n", sky_name);

  /* Load 6 PCX face textures and compose cubemap data */
  uint8_t* cubemap_data = NULL;
  uint32_t cube_size    = 0;
  if (!q2_load_sky_faces(&state.pak, sky_name, &cubemap_data, &cube_size)) {
    printf("[Q2] Failed to load sky faces — skybox disabled\n");
    return;
  }

  /* Create cubemap GPU texture */
  WGPUTextureDescriptor tex_desc = {
    .label     = STRVIEW("Q2 skybox cubemap"),
    .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension = WGPUTextureDimension_2D,
    .size = {.width = cube_size, .height = cube_size, .depthOrArrayLayers = 6},
    .format          = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount   = 1,
    .sampleCount     = 1,
    .viewFormatCount = 0,
  };
  state.skybox.cubemap_handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);
  ASSERT(state.skybox.cubemap_handle != NULL);

  /* Upload each face */
  uint32_t face_bytes = cube_size * cube_size * 4;
  for (uint32_t face = 0; face < 6; face++) {
    wgpuQueueWriteTexture(wgpu_context->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture  = state.skybox.cubemap_handle,
                            .mipLevel = 0,
                            .origin   = {.x = 0, .y = 0, .z = face},
                            .aspect   = WGPUTextureAspect_All,
                          },
                          cubemap_data + face * face_bytes, face_bytes,
                          &(WGPUTexelCopyBufferLayout){
                            .offset       = 0,
                            .bytesPerRow  = cube_size * 4,
                            .rowsPerImage = cube_size,
                          },
                          &(WGPUExtent3D){.width              = cube_size,
                                          .height             = cube_size,
                                          .depthOrArrayLayers = 1});
  }
  free(cubemap_data);

  /* Create cubemap texture view */
  state.skybox.cubemap_view = wgpuTextureCreateView(
    state.skybox.cubemap_handle, &(WGPUTextureViewDescriptor){
                                   .label  = STRVIEW("Q2 skybox cubemap view"),
                                   .format = WGPUTextureFormat_RGBA8Unorm,
                                   .dimension = WGPUTextureViewDimension_Cube,
                                   .baseMipLevel    = 0,
                                   .mipLevelCount   = 1,
                                   .baseArrayLayer  = 0,
                                   .arrayLayerCount = 6,
                                 });
  ASSERT(state.skybox.cubemap_view != NULL);

  /* Cubemap sampler (clamp-to-edge for seamless edges) */
  state.skybox.cubemap_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Q2 skybox sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.skybox.cubemap_sampler != NULL);

  /* Uniform buffer for inverse view-projection (rotation only) */
  state.skybox.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 skybox uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4),
                  });

  /* Bind group layout: uniform + sampler + cubemap texture */
  WGPUBindGroupLayoutEntry sky_entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(mat4),
      },
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_Cube,
      },
    },
  };
  state.skybox.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Q2 skybox bind group layout"),
                            .entryCount = 3,
                            .entries    = sky_entries,
                          });
  ASSERT(state.skybox.bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry sky_bg_entries[3] = {
    [0] = {.binding = 0,
           .buffer  = state.skybox.uniform_buffer.buffer,
           .offset  = 0,
           .size    = sizeof(mat4)},
    [1] = {.binding = 1, .sampler = state.skybox.cubemap_sampler},
    [2] = {.binding = 2, .textureView = state.skybox.cubemap_view},
  };
  state.skybox.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Q2 skybox bind group"),
                            .layout     = state.skybox.bind_group_layout,
                            .entryCount = 3,
                            .entries    = sky_bg_entries,
                          });
  ASSERT(state.skybox.bind_group != NULL);

  /* Pipeline layout */
  state.skybox.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Q2 skybox pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.skybox.bind_group_layout,
                          });
  ASSERT(state.skybox.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule sky_vs
    = wgpu_create_shader_module(wgpu_context->device, q2_skybox_vs_wgsl);
  WGPUShaderModule sky_fs
    = wgpu_create_shader_module(wgpu_context->device, q2_skybox_fs_wgsl);

  /* Render pipeline: fullscreen triangle, no vertex buffers, depth <= */
  WGPUDepthStencilState sky_depth
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = false, /* Don't write depth — skybox at infinity */
    });
  sky_depth.depthCompare = WGPUCompareFunction_LessEqual;

  state.skybox.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Q2 skybox pipeline"),
      .layout = state.skybox.pipeline_layout,
      .vertex = {
        .module      = sky_vs,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 0,
        .buffers     = NULL,
      },
      .fragment = &(WGPUFragmentState){
        .module      = sky_fs,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &sky_depth,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });
  ASSERT(state.skybox.pipeline != NULL);

  wgpuShaderModuleRelease(sky_vs);
  wgpuShaderModuleRelease(sky_fs);

  state.skybox.enabled = true;
  printf("[Q2] Skybox loaded: \"%s\" (%ux%u per face)\n", sky_name, cube_size,
         cube_size);
}

/* -------------------------------------------------------------------------- *
 * Camera Setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type = CameraType_FirstPerson;
  /* WebGPU uses Y-up NDC (same as OpenGL), not Vulkan's Y-down NDC.
   * flip_y=false means no projection Y-flip; the coordinate conversion
   * (Q2 Z-up → WebGPU Y-up) is done in the vertex data instead. */
  state.camera.flip_y    = false;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;

  /* Try to get spawn position from entity string */
  float spawn_pos[3] = {0};
  float spawn_angle  = 0.0f;

  if (state.bsp.entities
      && q2_find_player_start(state.bsp.entities, spawn_pos, &spawn_angle)) {
    /* Quake 2's DEFAULT_VIEWHEIGHT is 22 units above the player's waist
     * origin.  The BSP info_player_start origin is at the player origin, not
     * the eye, so we lift the camera by this amount in Q2's Z axis. */
    const float Q2_PLAYER_EYE_HEIGHT = 22.0f;

    /* Convert Q2 coords (X right, Y fwd, Z up) to render space (X right, Y up,
     * -Z fwd).  camera_set_position() expects (−render_X, +render_Y, −render_Z)
     * because it stores the negative world position for the view-matrix
     * translation.  render = (Q2_x, Q2_z, −Q2_y), so input is:
     *   (-Q2_x,  Q2_z + eye_height,  Q2_y) */
    camera_set_position(
      &state.camera,
      (vec3){-spawn_pos[0], spawn_pos[2] + Q2_PLAYER_EYE_HEIGHT, spawn_pos[1]});
    camera_set_rotation(&state.camera,
                        (vec3){0.0f, -(spawn_angle - 90.0f), 0.0f});
    printf("[Q2] Spawn: Q2=(%.0f, %.0f, %.0f) angle=%.0f [eye at Q2_Z=%.0f]\n",
           spawn_pos[0], spawn_pos[1], spawn_pos[2], spawn_angle,
           spawn_pos[2] + Q2_PLAYER_EYE_HEIGHT);
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
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Quake 2 Renderer", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* --- Map Selection --- */
  if (state.num_maps > 1) {
    igTextColored((ImVec4){0.4f, 1.0f, 0.4f, 1.0f}, "Map Selection");

    /* Build preview string for current map */
    const char* preview = (state.current_map_index >= 0) ?
                            state.maps[state.current_map_index].display_name :
                            "(none)";

    if (igBeginCombo("##map_combo", preview, ImGuiComboFlags_None)) {
      for (uint32_t i = 0; i < state.num_maps; i++) {
        bool is_selected = ((int32_t)i == state.current_map_index);

        /* Format: "Display Name (filename.bsp)" */
        char label[128];
        /* Extract just filename from pak_path (e.g. "base1.bsp" from
         * "maps/base1.bsp") */
        const char* slash = strrchr(state.maps[i].pak_path, '/');
        const char* fname = slash ? (slash + 1) : state.maps[i].pak_path;
        snprintf(label, sizeof(label), "%s (%s)", state.maps[i].display_name,
                 fname);

        if (igSelectable(label, is_selected, ImGuiSelectableFlags_None,
                         (ImVec2){0, 0})) {
          if ((int32_t)i != state.current_map_index) {
            state.requested_map_index = (int32_t)i;
          }
        }

        /* Tooltip with description */
        if (igIsItemHovered(ImGuiHoveredFlags_None)
            && state.maps[i].description[0] != '\0') {
          igSetTooltip("%s", state.maps[i].description);
        }

        if (is_selected) {
          igSetItemDefaultFocus();
        }
      }
      igEndCombo();
    }
    igSeparator();
  }

  /* --- Performance --- */
  igTextColored((ImVec4){1.0f, 0.85f, 0.0f, 1.0f}, "Performance");
  igText("Resolution : %d x %d", wgpu_context->width, wgpu_context->height);
  igText("FPS        : %.1f", state.fps);
  igText("Frame time : %.2f ms", state.frame_time_ms);
  igSeparator();

  /* --- Geometry --- */
  igTextColored((ImVec4){0.6f, 0.9f, 1.0f, 1.0f}, "Geometry");
  igText("Faces      : %u / %u (vis/total)", state.total_vertices / 3,
         state.master_vert_count / 3);
  igText("Vertices   : %u / %u (vis/total)", state.total_vertices,
         state.master_vert_count);
  igText("Textures   : %u", state.num_textures);
  igSeparator();

  /* --- BSP / PVS --- */
  igTextColored((ImVec4){0.6f, 1.0f, 0.7f, 1.0f}, "BSP / PVS");
  igText("Leaf       : %d", state.current_leaf);
  if (state.current_leaf >= 0
      && (uint32_t)state.current_leaf < state.bsp.num_leaves) {
    uint16_t cluster = state.bsp.leaves[state.current_leaf].cluster;
    igText("Cluster    : %u / %u", (uint32_t)cluster, state.num_pvs_clusters);
  }
  igSeparator();

  /* --- Camera --- */
  igTextColored((ImVec4){1.0f, 0.7f, 0.5f, 1.0f}, "Camera");
  igText("Position   : (%.0f, %.0f, %.0f)", -state.camera.position[0],
         state.camera.position[1], -state.camera.position[2]);
  igText("Rotation   : (%.1f, %.1f)", state.camera.rotation[0],
         state.camera.rotation[1]);
  igSeparator();

  igTextDisabled("WASD = move  |  Mouse = look");
  igSeparator();

  /* --- Render Settings --- */
  igTextColored((ImVec4){1.0f, 1.0f, 0.6f, 1.0f}, "Render Settings");
  igCheckbox("Wireframe", &state.show_wireframe);
  if (state.skybox.cubemap_handle) {
    igCheckbox("Skybox", &state.skybox.enabled);
  }
  else {
    igTextDisabled("Skybox  (not available)");
  }
  igSeparator();

  /* --- View Mode --- */
  igTextColored((ImVec4){1.0f, 1.0f, 0.6f, 1.0f}, "View Mode");
  {
    int vm = (int)state.debug.view_mode;
    igRadioButtonIntPtr("Normal (textured)", &vm, Q2_VIEW_NORMAL);
    igRadioButtonIntPtr("Colored (non-textured)", &vm, Q2_VIEW_COLORED);
    igRadioButtonIntPtr("Collision volumes", &vm, Q2_VIEW_COLLISION);
    state.debug.view_mode = (q2_view_mode_t)vm;
  }
  igSeparator();

  /* --- Gizmo --- */
  igCheckbox("Show Gizmo (origin axes)", &state.debug.show_gizmo);
  igSeparator();

  /* --- MD2 Models --- */
  if (state.md2.enabled) {
    igTextColored((ImVec4){0.9f, 0.6f, 1.0f, 1.0f}, "MD2 Models");
    igCheckbox("Show Models", &state.md2.show_models);
    igText("Models     : %u", state.md2.num_models);
    igText("Entities   : %u", state.md2.num_entities);
    igText("Visible    : %u", state.md2.num_visible);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * MD2 Model System Initialization
 *
 * Loads MD2 models referenced by BSP entities, creates GPU textures and
 * rendering pipeline for animated model display.
 * -------------------------------------------------------------------------- */

/**
 * @brief Create the MD2 rendering pipeline (bind group layouts, pipeline).
 */
static void init_md2_pipeline(wgpu_context_t* wgpu_context)
{
  /* Sampler for MD2 textures */
  state.md2.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("MD2 sampler"),
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

  /* Group 0: Shared (uniform buffer with dynamic offset + sampler) */
  WGPUBindGroupLayoutEntry shared_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = true,
        .minBindingSize   = sizeof(q2_md2_uniforms_t),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };
  state.md2.bg_layout_shared = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("MD2 shared BGL"),
                            .entryCount = (uint32_t)ARRAY_SIZE(shared_entries),
                            .entries    = shared_entries,
                          });

  /* Group 1: Per-model texture */
  WGPUBindGroupLayoutEntry tex_entry = (WGPUBindGroupLayoutEntry){
    .binding    = 0,
    .visibility = WGPUShaderStage_Fragment,
    .texture = (WGPUTextureBindingLayout){
      .sampleType    = WGPUTextureSampleType_Float,
      .viewDimension = WGPUTextureViewDimension_2D,
    },
  };
  state.md2.bg_layout_texture = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("MD2 texture BGL"),
                            .entryCount = 1,
                            .entries    = &tex_entry,
                          });

  /* Query minimum uniform buffer offset alignment */
  {
    WGPULimits limits = {0};
    wgpuDeviceGetLimits(wgpu_context->device, &limits);
    uint32_t min_align = limits.minUniformBufferOffsetAlignment;
    /* Round up sizeof(q2_md2_uniforms_t) to alignment boundary */
    state.md2.uniform_alignment
      = ((uint32_t)sizeof(q2_md2_uniforms_t) + min_align - 1) / min_align
        * min_align;
  }

  /* Dynamic uniform buffer (sized for all entities, grown as needed) */
  uint32_t initial_ub_cap  = Q2_MD2_MAX_ENTITIES * state.md2.uniform_alignment;
  state.md2.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "MD2 dynamic uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = initial_ub_cap,
                  });
  state.md2.ub_capacity = initial_ub_cap;

  /* Shared bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.md2.uniform_buffer.buffer,
      .offset  = 0,
      .size    = sizeof(q2_md2_uniforms_t),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.md2.sampler,
    },
  };
  state.md2.bg_shared = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("MD2 shared BG"),
                            .layout     = state.md2.bg_layout_shared,
                            .entryCount = 2,
                            .entries    = bg_entries,
                          });

  /* Pipeline layout */
  WGPUBindGroupLayout layouts[2]
    = {state.md2.bg_layout_shared, state.md2.bg_layout_texture};
  state.md2.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("MD2 pipeline layout"),
                            .bindGroupLayoutCount = 2,
                            .bindGroupLayouts     = layouts,
                          });

  /* Shader modules */
  WGPUShaderModule vs
    = wgpu_create_shader_module(wgpu_context->device, q2_md2_vs_wgsl);
  WGPUShaderModule fs
    = wgpu_create_shader_module(wgpu_context->device, q2_md2_fs_wgsl);

  /* Vertex buffer layout: pos(3) + next_pos(3) + tex_uv(2) */
  WGPU_VERTEX_BUFFER_LAYOUT(
    md2_vertex, sizeof(q2_md2_render_vertex_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       offsetof(q2_md2_render_vertex_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(q2_md2_render_vertex_t, next_pos)),
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       offsetof(q2_md2_render_vertex_t, tex_uv)))

  WGPUBlendState blend = wgpu_create_blend_state(true);

  WGPUDepthStencilState depth
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth.depthCompare = WGPUCompareFunction_Less;

  state.md2.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("MD2 render pipeline"),
      .layout = state.md2.pipeline_layout,
      .vertex = {
        .module      = vs,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &md2_vertex_vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fs,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &depth,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });

  wgpuShaderModuleRelease(vs);
  wgpuShaderModuleRelease(fs);
}

/**
 * @brief Load all MD2 models referenced by BSP entities.
 *
 * Parses the entity string, finds unique model paths, loads MD2 data and
 * skin textures, and creates GPU resources.
 */
static void init_md2_models(wgpu_context_t* wgpu_context)
{
  if (!state.bsp.entities) {
    return;
  }

  const char* model_paths[Q2_MD2_MAX_MODELS];
  const char* skin_overrides[Q2_MD2_MAX_MODELS];
  uint32_t model_count = 0;
  memset(skin_overrides, 0, sizeof(skin_overrides));

  /* Parse entities and get unique model paths + skin overrides */
  if (!q2_parse_md2_entities(state.bsp.entities, &state.bsp,
                             &state.md2.entities, &state.md2.num_entities,
                             model_paths, skin_overrides, &model_count)) {
    printf("[MD2] No MD2 entities found in BSP\n");
    return;
  }

  /* Load each unique MD2 model from PAK */
  uint32_t loaded = 0;
  for (uint32_t i = 0; i < model_count; i++) {
    const q2_pak_entry_t* entry = q2_pak_find(&state.pak, model_paths[i]);
    if (!entry) {
      printf("[MD2] Model not found in PAK: %s\n", model_paths[i]);
      continue;
    }

    uint32_t md2_size       = 0;
    const uint8_t* md2_data = q2_pak_get_data(&state.pak, entry, &md2_size);
    if (!md2_data) {
      continue;
    }

    if (!q2_md2_parse(md2_data, md2_size, &state.md2.models[i].model)) {
      printf("[MD2] Failed to parse: %s\n", model_paths[i]);
      continue;
    }

    /* Load skin texture (skin_override has highest priority) */
    uint8_t* rgba  = NULL;
    uint32_t tex_w = 0, tex_h = 0;
    if (q2_md2_load_skin(&state.pak, model_paths[i], &state.md2.models[i].model,
                         skin_overrides[i], &rgba, &tex_w, &tex_h)) {
      state.md2.models[i].gpu_tex = wgpu_create_texture(
        wgpu_context, &(wgpu_texture_desc_t){
                        .extent = (WGPUExtent3D){.width              = tex_w,
                                                 .height             = tex_h,
                                                 .depthOrArrayLayers = 1},
                        .format = WGPUTextureFormat_RGBA8Unorm,
                        .pixels = {.ptr = rgba, .size = tex_w * tex_h * 4},
                      });
      free(rgba);
    }
    else {
      /* Create 2x2 magenta placeholder */
      printf("[MD2] Skin not found for model: %s (skin_name: '%s')\n",
             model_paths[i], state.md2.models[i].model.skin_name);
      uint8_t placeholder[2 * 2 * 4];
      for (int p = 0; p < 4; p++) {
        placeholder[p * 4 + 0] = 255;
        placeholder[p * 4 + 1] = 0;
        placeholder[p * 4 + 2] = 255;
        placeholder[p * 4 + 3] = 255;
      }
      state.md2.models[i].gpu_tex = wgpu_create_texture(
        wgpu_context,
        &(wgpu_texture_desc_t){
          .extent
          = (WGPUExtent3D){.width = 2, .height = 2, .depthOrArrayLayers = 1},
          .format = WGPUTextureFormat_RGBA8Unorm,
          .pixels = {.ptr = placeholder, .size = sizeof(placeholder)},
        });
    }

    /* Create per-model texture bind group */
    WGPUBindGroupEntry tex_bg_entry = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.md2.models[i].gpu_tex.view,
    };
    state.md2.tex_bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("MD2 tex BG"),
                              .layout     = state.md2.bg_layout_texture,
                              .entryCount = 1,
                              .entries    = &tex_bg_entry,
                            });

    state.md2.models[i].loaded = true;
    loaded++;
  }

  state.md2.num_models      = model_count;
  state.md2.anim_start_time = stm_now();

  /* Initialize next_frame for entities now that models are loaded.
   * Gameplay-static entities stay on frame 0 (no animation). */
  for (uint32_t i = 0; i < state.md2.num_entities; i++) {
    q2_md2_entity_t* ent = &state.md2.entities[i];
    if (ent->is_gameplay_static) {
      ent->current_frame = 0;
      ent->next_frame    = 0;
      ent->interp        = 0.0f;
      continue;
    }
    if (ent->model_index >= 0 && (uint32_t)ent->model_index < model_count
        && state.md2.models[ent->model_index].loaded) {
      int32_t nf      = state.md2.models[ent->model_index].model.num_frames;
      ent->next_frame = (nf > 1) ? 1 : 0;
    }
  }

  /* Allocate visible entities array */
  state.md2.visible_entities
    = (uint32_t*)malloc(state.md2.num_entities * sizeof(uint32_t));
  state.md2.num_visible = 0;

  if (loaded > 0) {
    state.md2.enabled = true;
    printf("[MD2] Models ready: %u loaded, %u entities\n", loaded,
           state.md2.num_entities);
  }
}

/**
 * @brief Update MD2 visible entities based on current PVS cluster.
 *
 * Filters entities to only those in visible clusters or within cull distance.
 */
static void md2_update_visibility(const float cam_pos_q2[3],
                                  uint16_t current_cluster)
{
  state.md2.num_visible = 0;

  if (!state.md2.enabled || !state.md2.show_models) {
    return;
  }

  /* If we have PVS data, use cluster-based culling */
  bool use_pvs = (state.cluster_faces != NULL && state.num_pvs_clusters > 0
                  && current_cluster != Q2_CLUSTER_INVALID
                  && current_cluster < state.num_pvs_clusters);

  /* Decompress PVS for current cluster if available */
  uint8_t* vis_set = NULL;
  if (use_pvs && state.bsp.vis_data && state.bsp.vis_offsets) {
    vis_set                  = (uint8_t*)calloc(state.num_pvs_clusters, 1);
    vis_set[current_cluster] = 1; /* Always see own cluster */

    /* pvs_offset is relative to the visibility lump start.
     * lump_base = vis_offsets - sizeof(uint32_t) points to lump start.
     * This matches the approach used in q2_decompress_pvs(). */
    const uint8_t* lump_base
      = (const uint8_t*)state.bsp.vis_offsets - sizeof(uint32_t);
    uint32_t lump_size
      = sizeof(uint32_t)
        + state.num_pvs_clusters * (uint32_t)sizeof(q2_vis_offset_t)
        + state.bsp.vis_data_size;
    uint32_t v       = state.bsp.vis_offsets[current_cluster].pvs;
    uint32_t cluster = 0;
    while (cluster < state.num_pvs_clusters && v < lump_size) {
      uint8_t byte = lump_base[v];
      if (byte == 0) {
        v++;
        if (v >= lump_size) {
          break;
        }
        cluster += 8 * (uint32_t)lump_base[v];
      }
      else {
        for (int bit = 0; bit < 8 && cluster < state.num_pvs_clusters;
             bit++, cluster++) {
          if (byte & (1 << bit)) {
            vis_set[cluster] = 1;
          }
        }
      }
      v++;
    }
  }

  for (uint32_t i = 0; i < state.md2.num_entities; i++) {
    const q2_md2_entity_t* ent = &state.md2.entities[i];

    if (ent->model_index < 0
        || (uint32_t)ent->model_index >= state.md2.num_models
        || !state.md2.models[ent->model_index].loaded) {
      continue;
    }

    /* PVS cluster check */
    if (vis_set && ent->cluster >= 0
        && (uint32_t)ent->cluster < state.num_pvs_clusters) {
      if (!vis_set[ent->cluster]) {
        continue; /* Entity not in visible cluster */
      }
    }

    /* Distance cull */
    float dx   = ent->origin[0] - cam_pos_q2[0];
    float dy   = ent->origin[1] - cam_pos_q2[1];
    float dz   = ent->origin[2] - cam_pos_q2[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist > Q2_MD2_CULL_DISTANCE) {
      continue;
    }

    state.md2.visible_entities[state.md2.num_visible++] = i;
  }

  free(vis_set);
}

/* -------------------------------------------------------------------------- *
 * Map Switching
 *
 * Cleanly releases all map-specific GPU and CPU resources, then loads and
 * initialises a new BSP map from the cached PAK data.
 * -------------------------------------------------------------------------- */

/* Forward declarations for init helpers used by switch_map */
static void init_gpu_textures(wgpu_context_t* wgpu_context);
static void init_bind_groups(wgpu_context_t* wgpu_context);
static void init_debug_colored(wgpu_context_t* wgpu_context);
static void init_debug_brush_volumes(wgpu_context_t* wgpu_context);
static void init_skybox(wgpu_context_t* wgpu_context);
static void init_md2_models(wgpu_context_t* wgpu_context);
static void init_camera(wgpu_context_t* wgpu_context);

/**
 * @brief Release all map-specific resources (GPU + CPU).
 *
 * Keeps the PAK archive in memory, GPU pipelines/layouts, sampler, and other
 * resources that are shared across maps. Only releases resources that change
 * when switching BSP maps.
 */
static void cleanup_map_resources(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Wait for GPU to finish any pending work */
  if (wgpu_context && wgpu_context->device) {
    wgpuDeviceTick(wgpu_context->device);
  }

  /* --- Release map-specific bind groups (not layouts) --- */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_shared)
  state.bg_shared = NULL;

  /* --- Release per-texture GPU resources --- */
  for (uint32_t t = 0; t < state.num_textures; t++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.textures[t].bind_group)
    state.textures[t].bind_group = NULL;
    wgpu_destroy_texture(&state.textures[t].gpu_tex);
  }
  state.num_textures = 0;

  /* --- Release lightmap atlas --- */
  wgpu_destroy_texture(&state.lightmap_tex);

  /* --- Release vertex buffer (will be recreated for new map) --- */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer.buffer)
  state.vertex_buffer.buffer = NULL;
  state.vertex_buffer.size   = 0;

  /* --- Release skybox resources --- */
  if (state.skybox.enabled || state.skybox.cubemap_handle) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox.pipeline)
    state.skybox.pipeline = NULL;
    WGPU_RELEASE_RESOURCE(PipelineLayout, state.skybox.pipeline_layout)
    state.skybox.pipeline_layout = NULL;
    WGPU_RELEASE_RESOURCE(BindGroup, state.skybox.bind_group)
    state.skybox.bind_group = NULL;
    WGPU_RELEASE_RESOURCE(BindGroupLayout, state.skybox.bind_group_layout)
    state.skybox.bind_group_layout = NULL;
    WGPU_RELEASE_RESOURCE(Buffer, state.skybox.uniform_buffer.buffer)
    state.skybox.uniform_buffer.buffer = NULL;
    WGPU_RELEASE_RESOURCE(Sampler, state.skybox.cubemap_sampler)
    state.skybox.cubemap_sampler = NULL;
    WGPU_RELEASE_RESOURCE(TextureView, state.skybox.cubemap_view)
    state.skybox.cubemap_view = NULL;
    WGPU_RELEASE_RESOURCE(Texture, state.skybox.cubemap_handle)
    state.skybox.cubemap_handle = NULL;
    state.skybox.enabled        = false;
  }

  /* --- Release debug geometry buffers (map-dependent data) --- */
  WGPU_RELEASE_RESOURCE(Buffer, state.debug.colored_vb.buffer)
  state.debug.colored_vb.buffer  = NULL;
  state.debug.colored_vert_count = 0;
  WGPU_RELEASE_RESOURCE(Buffer, state.debug.brush_vb.buffer)
  state.debug.brush_vb.buffer  = NULL;
  state.debug.brush_vert_count = 0;

  /* --- Release MD2 model resources (per-map data only) --- */
  if (state.md2.enabled) {
    for (uint32_t i = 0; i < state.md2.num_models; i++) {
      WGPU_RELEASE_RESOURCE(BindGroup, state.md2.tex_bind_groups[i])
      state.md2.tex_bind_groups[i] = NULL;
      wgpu_destroy_texture(&state.md2.models[i].gpu_tex);
      q2_md2_destroy(&state.md2.models[i].model);
      state.md2.models[i].loaded = false;
    }
    state.md2.num_models = 0;

    free(state.md2.entities);
    state.md2.entities     = NULL;
    state.md2.num_entities = 0;
    free(state.md2.visible_entities);
    state.md2.visible_entities = NULL;
    state.md2.num_visible      = 0;
    state.md2.num_draw_cmds    = 0;
    state.md2.enabled          = false;
    /* Note: bg_shared, uniform_buffer, vertex_buffer, vb_staging, ub_staging
     * are kept alive — they are shared infrastructure resized as needed. */
  }

  /* --- Free PVS data --- */
  q2_cluster_faces_destroy(state.cluster_faces, state.num_pvs_clusters);
  state.cluster_faces    = NULL;
  state.num_pvs_clusters = 0;

  /* --- Free per-face mesh data --- */
  free(state.face_meshes);
  state.face_meshes = NULL;

  /* --- Free master vertex array --- */
  free(state.master_vertices);
  state.master_vertices   = NULL;
  state.master_vert_count = 0;
  state.total_vertices    = 0;

  /* --- Free BSP data --- */
  q2_bsp_destroy(&state.bsp);

  /* Reset BSP leaf tracking to force PVS rebuild */
  state.current_leaf = -1;
  state.prev_leaf    = -2;
}

/**
 * @brief Load and initialise a BSP map by index into state.maps[].
 *
 * Cleans up the current map (if any) and loads the new one from the cached
 * PAK data. Recreates all map-specific GPU resources.
 *
 * @param wgpu_context  WebGPU context.
 * @param map_index     Index into state.maps[].
 * @return true on success.
 */
static bool switch_map(wgpu_context_t* wgpu_context, int32_t map_index)
{
  if (map_index < 0 || (uint32_t)map_index >= state.num_maps) {
    fprintf(stderr, "[Q2] Invalid map index: %d\n", map_index);
    return false;
  }

  printf("\n[Q2] === Switching to map: %s (\"%s\") ===\n",
         state.maps[map_index].pak_path, state.maps[map_index].display_name);

  /* Clean up current map resources */
  if (state.current_map_index >= 0) {
    cleanup_map_resources(wgpu_context);
  }

  /* Parse BSP from cached PAK data */
  const q2_pak_entry_t* bsp_entry
    = q2_pak_find(&state.pak, state.maps[map_index].pak_path);
  if (!bsp_entry) {
    fprintf(stderr, "[ERROR] BSP not found in PAK: %s\n",
            state.maps[map_index].pak_path);
    return false;
  }

  uint32_t bsp_size       = 0;
  const uint8_t* bsp_data = q2_pak_get_data(&state.pak, bsp_entry, &bsp_size);
  if (!bsp_data || !q2_bsp_parse(bsp_data, bsp_size, &state.bsp)) {
    fprintf(stderr, "[ERROR] BSP parsing failed: %s\n",
            state.maps[map_index].pak_path);
    return false;
  }

  printf("[Q2] Loaded BSP: %s (%u faces, %u texinfos, %u vertices)\n",
         bsp_entry->filename, state.bsp.num_faces, state.bsp.num_texinfos,
         state.bsp.num_vertices);

  /* Build lightmap atlas */
  lm_atlas_t* atlas = (lm_atlas_t*)calloc(1, sizeof(lm_atlas_t));
  if (!atlas) {
    return false;
  }
  lm_atlas_init(atlas);

  /* Build triangulated geometry */
  q2_render_vertex_t* verts
    = build_map_geometry(&state.bsp, &state.pak, atlas, &state.total_vertices);
  if (!verts || state.total_vertices == 0) {
    free(atlas);
    return false;
  }

  printf("[Q2] Geometry: %u master vertices (%.1f KB)\n",
         state.master_vert_count,
         state.master_vert_count * sizeof(q2_render_vertex_t) / 1024.0);

  /* Pre-compute PVS face lists */
  state.cluster_faces = q2_precompute_all_pvs(&state.bsp, state.face_meshes,
                                              &state.num_pvs_clusters);

  /* Create vertex buffer */
  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Q2 vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = state.master_vert_count * sizeof(q2_render_vertex_t),
      .initial.data = verts,
    });

  /* Initial rebuild: sort all renderable faces into batch order */
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

  /* Upload lightmap atlas */
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
  free(atlas);

  /* Load WAL textures */
  init_gpu_textures(wgpu_context);

  /* Recreate bind groups (lightmap texture changed) */
  init_bind_groups(wgpu_context);

  /* Recreate debug geometry for new map */
  init_debug_colored(wgpu_context);
  init_debug_brush_volumes(wgpu_context);

  /* Load skybox for new map */
  init_skybox(wgpu_context);

  /* Load MD2 models for new map */
  init_md2_models(wgpu_context);

  /* Set camera to spawn position */
  init_camera(wgpu_context);

  state.current_map_index = map_index;
  printf("[Q2] Map switch complete: %s\n", state.maps[map_index].display_name);
  return true;
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

  /* Load PAK archive into memory (serves as cache for all map switches) and
   * discover all available BSP maps with metadata. */
  if (!load_pak_and_discover_maps(NULL)) {
    return EXIT_FAILURE;
  }

  /* --- Create shared GPU resources (persist across map switches) --- */

  /* Uniform buffer (MVP matrix + render flags) */
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Q2 uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(q2_uniforms_t),
                  });

  /* Sampler, bind group layouts, render pipelines */
  init_sampler(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_pipeline(wgpu_context);

  /* Debug / visualization pipelines (shared across maps)
   * NOTE: must come AFTER init_pipeline and uniform_buffer creation */
  init_debug_pipelines(wgpu_context);
  init_gizmo(wgpu_context);

  /* MD2 pipeline (shared across maps) */
  init_md2_pipeline(wgpu_context);

  /* ImGui overlay */
  imgui_overlay_init(wgpu_context);

  /* --- Load the default map --- */
  int32_t default_map = find_default_map_index();
  if (!switch_map(wgpu_context, default_map)) {
    return EXIT_FAILURE;
  }

  state.initialized = true;
  printf("[Q2] Renderer initialized successfully\n");
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* --- Handle pending PAK file reload (from drag & drop) --- */
  if (state.pak_reload.pending) {
    state.pak_reload.pending = false;
    if (!reload_pak(wgpu_context, state.pak_reload.path)) {
      fprintf(stderr, "[Q2] PAK reload failed, renderer may be in bad state\n");
      return EXIT_FAILURE;
    }
  }

  /* --- Handle pending map switch request from GUI --- */
  if (state.requested_map_index >= 0
      && state.requested_map_index != state.current_map_index) {
    int32_t req               = state.requested_map_index;
    state.requested_map_index = -1; /* Consume the request */
    if (!switch_map(wgpu_context, req)) {
      fprintf(stderr, "[Q2] Map switch failed, keeping current map\n");
    }
  }

  /* Delta time */
  uint64_t now          = stm_now();
  float dt              = (state.last_frame_time == 0) ?
                            (1.0f / 60.0f) :
                            (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Performance stats: smooth FPS averaged over 0.5-second windows */
  state.frame_time_ms = dt * 1000.0f;
  state.fps_frame_count++;
  if (state.fps_accum_start == 0) {
    state.fps_accum_start = now;
  }
  double fps_elapsed = stm_sec(stm_diff(now, state.fps_accum_start));
  if (fps_elapsed >= 0.5) {
    state.fps             = (float)(state.fps_frame_count / fps_elapsed);
    state.fps_frame_count = 0;
    state.fps_accum_start = now;
  }

  /* Update camera */
  camera_update(&state.camera, dt);

  /* --- BSP traversal: find current leaf and update VBO via PVS --- */
  if (state.cluster_faces && state.num_pvs_clusters > 0) {
    /* Convert stored camera position back to Q2 coordinate space for BSP leaf
     * traversal. camera_set_position() stores camera->position as the negative
     * world position needed for the view-matrix translation:
     *   position[0] = -Q2_x
     *   position[1] = -Q2_z  (Q2 height; negated by camera_set_position)
     *   position[2] =  Q2_y  (Q2 forward/backward)
     * Recover Q2 coords by inverting those relationships: */
    float q2_pos[3] = {
      -state.camera.position[0], /* Q2 X */
      state.camera.position[2],  /* Q2 Y (forward) */
      -state.camera.position[1], /* Q2 Z (height; stored negated) */
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

  /* Compute and upload uniforms (MVP matrix + render flags) */
  q2_uniforms_t uniforms = {0};
  glm_mat4_mul(state.camera.matrices.perspective, state.camera.matrices.view,
               uniforms.mvp);
  uniforms.wireframe_mode = state.show_wireframe ? 1u : 0u;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &uniforms, sizeof(q2_uniforms_t));

  /* Skybox: compute inverse of rotation-only view-projection matrix.
   * Strip translation from the view matrix so the skybox follows camera
   * rotation but stays at infinity. */
  if (state.skybox.enabled) {
    mat4 view_rot;
    glm_mat4_copy(state.camera.matrices.view, view_rot);
    /* Zero out translation (column 3, rows 0-2) */
    view_rot[3][0] = 0.0f;
    view_rot[3][1] = 0.0f;
    view_rot[3][2] = 0.0f;

    mat4 vp_rot, vp_rot_inv;
    glm_mat4_mul(state.camera.matrices.perspective, view_rot, vp_rot);
    glm_mat4_inv(vp_rot, vp_rot_inv);
    wgpuQueueWriteBuffer(wgpu_context->queue,
                         state.skybox.uniform_buffer.buffer, 0, &vp_rot_inv,
                         sizeof(mat4));
  }

  /* --- MD2 animation update --- */
  if (state.md2.enabled && state.md2.show_models) {
    /* Advance animation only for gameplay-animated entities.
     * Gameplay-static entities (barrels, items, weapons) stay on frame 0
     * because their MD2 frames are event animations (explosion, pickup),
     * not idle loops. */
    for (uint32_t i = 0; i < state.md2.num_entities; i++) {
      q2_md2_entity_t* ent = &state.md2.entities[i];
      if (ent->is_gameplay_static) {
        continue; /* Static object — keep frame 0, no animation */
      }
      if (ent->model_index < 0
          || (uint32_t)ent->model_index >= state.md2.num_models
          || !state.md2.models[ent->model_index].loaded) {
        continue;
      }
      const q2_md2_model_t* mdl = &state.md2.models[ent->model_index].model;
      if (mdl->num_frames <= 1) {
        continue; /* Single-frame model, no animation */
      }

      ent->interp += dt * Q2_MD2_ANIM_FPS;
      while (ent->interp >= 1.0f) {
        ent->interp -= 1.0f;
        ent->current_frame = ent->next_frame;
        ent->next_frame    = (ent->current_frame + 1) % mdl->num_frames;
      }
    }

    /* Update PVS visibility */
    if (state.cluster_faces && state.num_pvs_clusters > 0
        && state.current_leaf >= 0
        && (uint32_t)state.current_leaf < state.bsp.num_leaves) {
      float q2_cam[3] = {
        -state.camera.position[0],
        state.camera.position[2],
        -state.camera.position[1],
      };
      uint16_t cluster = state.bsp.leaves[state.current_leaf].cluster;
      md2_update_visibility(q2_cam, cluster);
    }
    else {
      /* No PVS — show all entities */
      state.md2.num_visible = 0;
      for (uint32_t i = 0; i < state.md2.num_entities; i++) {
        const q2_md2_entity_t* ent = &state.md2.entities[i];
        if (ent->model_index >= 0
            && (uint32_t)ent->model_index < state.md2.num_models
            && state.md2.models[ent->model_index].loaded) {
          state.md2.visible_entities[state.md2.num_visible++] = i;
        }
      }
    }
  }

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* --- Pre-compute MD2 draw commands (before render pass) --- */
  state.md2.num_draw_cmds = 0;
  if (state.md2.enabled && state.md2.show_models && state.md2.num_visible > 0) {
    uint32_t total_vb_bytes = 0;
    uint32_t total_ub_bytes = 0;
    const uint32_t align    = state.md2.uniform_alignment;

    /* Temp array to hold allocated vertex pointers until copy */
    q2_md2_render_vertex_t* tmp_verts[Q2_MD2_MAX_ENTITIES];

    for (uint32_t vi = 0; vi < state.md2.num_visible; vi++) {
      uint32_t ei                = state.md2.visible_entities[vi];
      const q2_md2_entity_t* ent = &state.md2.entities[ei];
      const q2_md2_model_t* mdl  = &state.md2.models[ent->model_index].model;

      q2_md2_render_vertex_t* verts = NULL;
      uint32_t vert_count           = 0;
      q2_md2_build_vertices(mdl, ent->current_frame, ent->next_frame, &verts,
                            &vert_count);
      if (!verts || vert_count == 0) {
        free(verts);
        continue;
      }

      if (state.md2.num_draw_cmds >= Q2_MD2_MAX_ENTITIES) {
        free(verts);
        break; /* Hit draw command limit */
      }

      uint32_t ci      = state.md2.num_draw_cmds;
      uint32_t vb_size = vert_count * (uint32_t)sizeof(q2_md2_render_vertex_t);

      state.md2.draw_cmds[ci].vb_offset    = total_vb_bytes;
      state.md2.draw_cmds[ci].vert_count   = vert_count;
      state.md2.draw_cmds[ci].ub_offset    = total_ub_bytes;
      state.md2.draw_cmds[ci].model_index  = ent->model_index;
      state.md2.draw_cmds[ci].entity_index = ei;
      tmp_verts[ci]                        = verts;

      total_vb_bytes += vb_size;
      total_ub_bytes += align;
      state.md2.num_draw_cmds++;
    }

    if (state.md2.num_draw_cmds > 0) {
      /* Grow GPU vertex buffer if needed */
      if (total_vb_bytes > state.md2.vb_capacity
          || !state.md2.vertex_buffer.buffer) {
        if (state.md2.vertex_buffer.buffer) {
          WGPU_RELEASE_RESOURCE(Buffer, state.md2.vertex_buffer.buffer)
        }
        state.md2.vertex_buffer = wgpu_create_buffer(
          wgpu_context,
          &(wgpu_buffer_desc_t){
            .label = "MD2 VB",
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
            .size  = total_vb_bytes,
          });
        state.md2.vb_capacity = total_vb_bytes;
      }

      /* Grow staging buffers if needed */
      if (total_vb_bytes > state.md2.vb_staging_cap) {
        state.md2.vb_staging = realloc(state.md2.vb_staging, total_vb_bytes);
        state.md2.vb_staging_cap = total_vb_bytes;
      }
      if (total_ub_bytes > state.md2.ub_staging_cap) {
        state.md2.ub_staging = realloc(state.md2.ub_staging, total_ub_bytes);
        state.md2.ub_staging_cap = total_ub_bytes;
      }

      /* Grow GPU uniform buffer if needed (and recreate bind group) */
      if (total_ub_bytes > state.md2.ub_capacity) {
        WGPU_RELEASE_RESOURCE(Buffer, state.md2.uniform_buffer.buffer)
        state.md2.uniform_buffer = wgpu_create_buffer(
          wgpu_context,
          &(wgpu_buffer_desc_t){
            .label = "MD2 dynamic uniform buffer",
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
            .size  = total_ub_bytes,
          });
        state.md2.ub_capacity = total_ub_bytes;

        /* Recreate bind group with new buffer */
        WGPU_RELEASE_RESOURCE(BindGroup, state.md2.bg_shared)
        WGPUBindGroupEntry bg_entries[2] = {
          [0] = (WGPUBindGroupEntry){
            .binding = 0,
            .buffer  = state.md2.uniform_buffer.buffer,
            .offset  = 0,
            .size    = sizeof(q2_md2_uniforms_t),
          },
          [1] = (WGPUBindGroupEntry){
            .binding = 1,
            .sampler = state.md2.sampler,
          },
        };
        state.md2.bg_shared = wgpuDeviceCreateBindGroup(
          wgpu_context->device, &(WGPUBindGroupDescriptor){
                                  .label      = STRVIEW("MD2 shared BG"),
                                  .layout     = state.md2.bg_layout_shared,
                                  .entryCount = 2,
                                  .entries    = bg_entries,
                                });
      }
      memset(state.md2.ub_staging, 0, total_ub_bytes);

      /* View-projection matrix (shared across all entities) */
      mat4 vp;
      glm_mat4_mul(state.camera.matrices.perspective,
                   state.camera.matrices.view, vp);

      /* Build staging data for each draw command */
      for (uint32_t i = 0; i < state.md2.num_draw_cmds; i++) {
        /* Copy vertex data into VB staging */
        uint32_t vb_size = state.md2.draw_cmds[i].vert_count
                           * (uint32_t)sizeof(q2_md2_render_vertex_t);
        memcpy(state.md2.vb_staging + state.md2.draw_cmds[i].vb_offset,
               tmp_verts[i], vb_size);
        free(tmp_verts[i]);

        const q2_md2_entity_t* ent
          = &state.md2.entities[state.md2.draw_cmds[i].entity_index];

        /* Build model matrix: Q2 coords -> render coords */
        mat4 model_mat;
        glm_mat4_identity(model_mat);
        vec3 render_pos = {
          ent->origin[0],
          ent->origin[2],
          -ent->origin[1],
        };
        glm_translate(model_mat, render_pos);
        float yaw_rad = glm_rad(ent->angle);
        glm_rotate_y(model_mat, yaw_rad, model_mat);
        mat4 coord_swap  = GLM_MAT4_IDENTITY_INIT;
        coord_swap[0][0] = 1.0f;
        coord_swap[0][1] = 0.0f;
        coord_swap[0][2] = 0.0f;
        coord_swap[1][0] = 0.0f;
        coord_swap[1][1] = 0.0f;
        coord_swap[1][2] = -1.0f;
        coord_swap[2][0] = 0.0f;
        coord_swap[2][1] = 1.0f;
        coord_swap[2][2] = 0.0f;
        glm_mat4_mul(model_mat, coord_swap, model_mat);

        q2_md2_uniforms_t md2_uniform = {0};
        glm_mat4_mul(vp, model_mat, md2_uniform.model_mvp);
        md2_uniform.interpolation = ent->interp;

        memcpy(state.md2.ub_staging + state.md2.draw_cmds[i].ub_offset,
               &md2_uniform, sizeof(md2_uniform));
      }

      /* Single GPU writes for all entities */
      wgpuQueueWriteBuffer(wgpu_context->queue, state.md2.vertex_buffer.buffer,
                           0, state.md2.vb_staging, total_vb_bytes);
      wgpuQueueWriteBuffer(wgpu_context->queue, state.md2.uniform_buffer.buffer,
                           0, state.md2.ub_staging, total_ub_bytes);
    }
  }

  /* Update render pass attachments */
  state.color_att.view = wgpu_context->swapchain_view;
  state.depth_att.view = wgpu_context->depth_stencil_view;

  /* Create command encoder and render pass */
  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rp
    = wgpuCommandEncoderBeginRenderPass(enc, &state.rp_desc);

  /* --- Draw skybox first (at infinity, behind all geometry) --- */
  if (state.skybox.enabled) {
    wgpuRenderPassEncoderSetPipeline(rp, state.skybox.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.skybox.bind_group, 0, 0);
    wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0); /* Fullscreen triangle */
  }

  /* --- Draw map geometry --- */
  if (state.debug.view_mode == Q2_VIEW_NORMAL) {
    /* Normal mode: bind textured pipeline and draw per-texture batches */
    wgpuRenderPassEncoderSetPipeline(rp, state.pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.vertex_buffer.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bg_shared, 0, 0);
    for (uint32_t t = 0; t < state.num_textures; t++) {
      if (state.textures[t].vert_count <= 0) {
        continue;
      }
      wgpuRenderPassEncoderSetBindGroup(rp, 1, state.textures[t].bind_group, 0,
                                        0);
      wgpuRenderPassEncoderDraw(rp, (uint32_t)state.textures[t].vert_count, 1,
                                (uint32_t)state.textures[t].vert_offset, 0);
    }
  }
  else if (state.debug.view_mode == Q2_VIEW_COLORED
           && state.debug.colored_vert_count > 0) {
    /* Colored mode: draw every visible face in a random per-face colour */
    wgpuRenderPassEncoderSetPipeline(rp, state.debug.tri_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.debug.bg, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.debug.colored_vb.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rp, state.debug.colored_vert_count, 1, 0, 0);
  }
  else if (state.debug.view_mode == Q2_VIEW_COLLISION
           && state.debug.brush_vert_count > 0) {
    /* Collision mode: wireframe AABB overlay of BSP sub-models */
    wgpuRenderPassEncoderSetPipeline(rp, state.debug.line_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.debug.bg, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.debug.brush_vb.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rp, state.debug.brush_vert_count, 1, 0, 0);
  }

  /* --- Draw MD2 models --- */
  if (state.md2.enabled && state.md2.show_models
      && state.md2.num_draw_cmds > 0) {
    wgpuRenderPassEncoderSetPipeline(rp, state.md2.pipeline);

    for (uint32_t i = 0; i < state.md2.num_draw_cmds; i++) {
      /* Bind shared group (uniform + sampler) with dynamic UB offset */
      uint32_t ub_offset = state.md2.draw_cmds[i].ub_offset;
      wgpuRenderPassEncoderSetBindGroup(rp, 0, state.md2.bg_shared, 1,
                                        &ub_offset);

      /* Bind per-model texture */
      wgpuRenderPassEncoderSetBindGroup(
        rp, 1, state.md2.tex_bind_groups[state.md2.draw_cmds[i].model_index], 0,
        0);

      /* Set vertex buffer with offset for this entity's data */
      uint64_t vb_off = state.md2.draw_cmds[i].vb_offset;
      uint64_t vb_sz  = state.md2.draw_cmds[i].vert_count
                       * (uint64_t)sizeof(q2_md2_render_vertex_t);
      wgpuRenderPassEncoderSetVertexBuffer(
        rp, 0, state.md2.vertex_buffer.buffer, vb_off, vb_sz);
      wgpuRenderPassEncoderDraw(rp, state.md2.draw_cmds[i].vert_count, 1, 0, 0);
    }
  }

  /* Gizmo: three coloured axis lines drawn at world origin (always on top) */
  if (state.debug.show_gizmo) {
    wgpuRenderPassEncoderSetPipeline(rp, state.debug.line_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.debug.bg, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.debug.gizmo_vb.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rp, 6, 1, 0, 0);
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

  /* Skip camera/scene input when ImGui captures the mouse or keyboard */
  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }

  if (!imgui_overlay_want_capture_keyboard()) {
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

  /* Handle drag & drop of .pak files */
  if (input_event->type == INPUT_EVENT_TYPE_FILE_DROP) {
    for (int i = 0; i < input_event->drop_count; i++) {
      const char* path = input_event->drop_paths[i];
      if (path_is_pak(path)) {
        printf("[Q2] PAK file dropped: %s\n", path);
        state.pak_reload.pending = true;
        strncpy(state.pak_reload.path, path, sizeof(state.pak_reload.path) - 1);
        state.pak_reload.path[sizeof(state.pak_reload.path) - 1] = '\0';
        break; /* Only accept the first .pak file */
      }
      else {
        printf("[Q2] Unsupported file type dropped: %s\n", path);
      }
    }
  }
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  imgui_overlay_shutdown();

  /* Release all map-specific resources (textures, geometry, MD2 per-map, etc.)
   */
  cleanup_map_resources(wgpu_context);

  /* Release shared GPU resources that persist across map switches */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layout_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layout_texture)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)

  /* Debug shared pipelines and gizmo (not per-map) */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.debug.line_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.debug.tri_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.debug.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.debug.bg)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.debug.bgl)
  WGPU_RELEASE_RESOURCE(Buffer, state.debug.gizmo_vb.buffer)

  /* MD2 shared pipeline infrastructure */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.md2.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.md2.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.md2.bg_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.md2.bg_layout_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.md2.bg_layout_texture)
  WGPU_RELEASE_RESOURCE(Buffer, state.md2.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.md2.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.md2.sampler)
  free(state.md2.vb_staging);
  state.md2.vb_staging = NULL;
  free(state.md2.ub_staging);
  state.md2.ub_staging = NULL;

  /* Release PAK archive (the in-memory cache) */
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
    mvp            : mat4x4f,
    wireframe_mode : u32,
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
    @location(2) bary  : vec3f,
  };

  /* Barycentric coordinate per vertex within its triangle (TriangleList).
   * vertex_index % 3 cycles 0,1,2 for each triangle regardless of the
   * global draw offset, giving (1,0,0), (0,1,0), (0,0,1) per corner. */
  const BARY = array<vec3f, 3>(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0)
  );

  @vertex
  fn main(
    in : VertexInput,
    @builtin(vertex_index) vertex_index : u32
  ) -> VertexOutput {
    var out : VertexOutput;
    out.position = uniforms.mvp * vec4f(in.position, 1.0);
    out.tex_uv   = in.tex_uv;
    out.lm_uv    = in.lm_uv;
    out.bary     = BARY[vertex_index % 3u];
    return out;
  }
);

static const char* q2_fragment_shader_wgsl = CODE(
  struct Uniforms {
    mvp            : mat4x4f,
    wireframe_mode : u32,
  };

  @group(0) @binding(0) var<uniform> uniforms    : Uniforms;
  @group(0) @binding(1) var          tex_sampler : sampler;
  @group(0) @binding(2) var          lightmap_tex: texture_2d<f32>;
  @group(1) @binding(0) var          diffuse_tex : texture_2d<f32>;

  /* Wireframe edge width in barycentric space.  Larger = thicker lines. */
  const WIRE_THRESHOLD : f32 = 0.018;

  /* Wireframe overlay colour: bright cyan, fully opaque. */
  const WIRE_COLOR : vec4f = vec4f(0.0, 0.9, 1.0, 1.0);

  @fragment
  fn main(
    @location(0) tex_uv : vec2f,
    @location(1) lm_uv  : vec2f,
    @location(2) bary   : vec3f,
  ) -> @location(0) vec4f {
    /* In wireframe mode discard interior fragments; keep only edge pixels. */
    if (uniforms.wireframe_mode != 0u) {
      let d = min(min(bary.x, bary.y), bary.z);
      if (d > WIRE_THRESHOLD) { discard; }
      return WIRE_COLOR;
    }
    let diffuse = textureSample(diffuse_tex, tex_sampler, tex_uv);
    let light   = textureSample(lightmap_tex, tex_sampler, lm_uv);
    return vec4f(diffuse.rgb * light.rgb, diffuse.a);
  }
);

/* Skybox vertex shader: fullscreen triangle with cubemap direction output.
 * Generates a large triangle from vertex_index (0,1,2) that covers the screen.
 * The clip-space position is output both as the rasterization position and as
 * a direction vector that will be transformed by the inverse VP matrix in the
 * fragment shader to produce cubemap lookup directions. */
static const char* q2_skybox_vs_wgsl = CODE(
  @group(0) @binding(0) var<uniform> view_dir_proj_inv : mat4x4f;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) direction : vec4f,
  };

  @vertex
  fn main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    let pos = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    var out : VertexOutput;
    let p = pos[vi];
    out.position = vec4f(p, 1.0, 1.0);
    out.direction = vec4f(p, 1.0, 1.0);
    return out;
  }
);

/* Skybox fragment shader: samples a cubemap using the direction reconstructed
 * from the inverse view-rotation-projection matrix.  No gamma correction is
 * applied because the PCX palette colours are sRGB values stored in the same
 * RGBA8Unorm format as the diffuse WAL textures; both are sampled in the same
 * colour space and modulated together in the final composited frame. */
static const char* q2_skybox_fs_wgsl = CODE(
  @group(0) @binding(0) var<uniform> view_dir_proj_inv : mat4x4f;
  @group(0) @binding(1) var sky_sampler : sampler;
  @group(0) @binding(2) var sky_texture : texture_cube<f32>;

  @fragment
  fn main(@location(0) direction : vec4f) -> @location(0) vec4f {
    let t = view_dir_proj_inv * direction;
    let uvw = normalize(t.xyz / t.w);
    let color = textureSample(sky_texture, sky_sampler, uvw);
    return vec4f(color.rgb, 1.0);
  }
);

/* Debug geometry vertex shader: transforms a coloured vertex with the MVP
 * matrix and forwards the vertex colour to the fragment stage. */
static const char* q2_debug_vs_wgsl = CODE(
  @group(0) @binding(0) var<uniform> mvp : mat4x4f;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) color    : vec3f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       color    : vec3f,
  };

  @vertex
  fn main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = mvp * vec4f(in.position, 1.0);
    out.color    = in.color;
    return out;
  }
);

/* Debug geometry fragment shader: outputs the interpolated vertex colour. */
static const char* q2_debug_fs_wgsl = CODE(
  @fragment
  fn main(@location(0) color : vec3f) -> @location(0) vec4f {
    return vec4f(color, 1.0);
  }
);

/* -------------------------------------------------------------------------- *
 * MD2 Model Shaders
 *
 * Vertex shader interpolates between current and next frame positions using
 * a uniform interpolation factor. Fragment shader applies the model's diffuse
 * texture with basic directional lighting for visibility.
 * -------------------------------------------------------------------------- */

/* MD2 vertex shader: lerp between two frame positions and transform by MVP */
static const char* q2_md2_vs_wgsl = CODE(
  struct Uniforms {
    model_mvp     : mat4x4f,
    interpolation : f32,
  };

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;

  struct VertexInput {
    @location(0) pos      : vec3f,
    @location(1) next_pos : vec3f,
    @location(2) tex_uv   : vec2f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       tex_uv  : vec2f,
  };

  @vertex
  fn main(in : VertexInput) -> VertexOutput {
    let interpolated = mix(in.pos, in.next_pos, uniforms.interpolation);
    var out : VertexOutput;
    out.position = uniforms.model_mvp * vec4f(interpolated, 1.0);
    out.tex_uv   = in.tex_uv;
    return out;
  }
);

/* MD2 fragment shader: sample diffuse texture with simple lighting */
static const char* q2_md2_fs_wgsl = CODE(
  @group(0) @binding(1) var md2_sampler : sampler;
  @group(1) @binding(0) var md2_texture : texture_2d<f32>;

  @fragment
  fn main(@location(0) tex_uv : vec2f) -> @location(0) vec4f {
    let color = textureSample(md2_texture, md2_sampler, tex_uv);
    if (color.a < 0.1) { discard; }
    return color;
  }
);
// clang-format on
