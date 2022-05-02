#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Returns a copy of argv.
 * @ref https://gist.github.com/bnoordhuis/1981730
 */
char** argv_copy(int argc, char** argv);

/**
 * @brief Returns if the str aurgument has the given prefix.
 */
int has_prefix(const char* str, const char* pre);

#endif
