#include "file.h"

#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "macro.h"

int file_exists(const char* filename)
{
  /* try to open file to read */
  FILE* file;
  if ((file = fopen(filename, "r"))) {
    fclose(file);
    return 1;
  }
  return 0;
}

const char* get_filename_extension(const char* filename)
{
  const char* dot = strrchr(filename, '.');
  if (!dot || dot == filename) {
    return "";
  }
  return dot + 1;
}

int filename_has_extension(const char* filename, const char* extension)
{
  const char* filename_extension = get_filename_extension(filename);
  return strcmp(filename_extension, extension) == 0 ? 1 : 0;
}

void read_file(const char* filename, file_read_result_t* result,
               int is_text_file)
{
  ASSERT(filename && result);
  FILE* file = fopen(filename, "rb");
  if (file == NULL) {
    log_error("Unable to open file '%s'\n", filename);
    exit(1);
  }
  fseek(file, 0, SEEK_END);
  result->size = ftell(file);
  fseek(file, 0, SEEK_SET);

  result->data = malloc(result->size + (is_text_file == 0 ? 0 : 1));
  fread(result->data, 1, result->size, file);
  fclose(file);
  if (is_text_file != 0) {
    result->data[result->size] = 0;
  }
}
