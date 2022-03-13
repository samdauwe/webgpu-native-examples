#ifndef FILE_H
#define FILE_H

#include <stdint.h>

typedef struct file_read_result_t {
  uint32_t size;
  uint8_t* data;
} file_read_result_t;

/**
 * @brief Check if a file exist using fopen() function.
 * @param filename the name of the file
 * @return 1 if the file exist otherwise return 0
 */
int file_exists(const char* filename);

/**
 * @brief Returns the extension of the file.
 * @param filename the name of the file
 * @return extension of the file
 */
const char* get_filename_extension(const char* filename);

/**
 * @brief Check if a filename has the specified extension.
 * @param filename the name of the file
 * @param extension the file extension to check
 * @return 1 if the file exist otherwise return 0
 */
int filename_has_extension(const char* filename, const char* extension);

/**
 * @brief Reads the file with the specified filename and writes data and size to
 * 'result'.
 * @param filename the name of the file
 * @param result the file read result
 */
void read_file(const char* filename, file_read_result_t* result,
               int is_text_file);

#endif
