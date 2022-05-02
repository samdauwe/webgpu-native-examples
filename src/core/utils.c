#include "utils.h"

#include <stdlib.h>
#include <string.h>

char** argv_copy(int argc, char** argv)
{
  size_t strlen_sum;
  char** argp;
  char* data;
  size_t len;
  int i;

  strlen_sum = 0;
  for (i = 0; i < argc; i++)
    strlen_sum += strlen(argv[i]) + 1;

  argp = malloc(sizeof(char*) * (argc + 1) + strlen_sum);
  if (!argp)
    return NULL;
  data = (char*)argp + sizeof(char*) * (argc + 1);

  for (i = 0; i < argc; i++) {
    argp[i] = data;
    len     = strlen(argv[i]) + 1;
    memcpy(data, argv[i], len);
    data += len;
  }
  argp[argc] = NULL;

  return argp;
}

int has_prefix(const char* str, const char* pre)
{
  return strncmp(pre, str, strlen(pre)) == 0;
}
