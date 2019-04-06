#ifndef _NEWJOY_MATMAN_HPP_
#define _NEWJOY_MATMAN_HPP_

#include <stdio.h>
int allocate_array(float** a, int m, int n);
int allocate_int_array(int** a, int m, int n);
void display_array(char* name, float* a, int m, int n);
void display_int_array(char* name, int* a, int m, int n);
int read_array(FILE* file, float* a, int m, int n);
void free_array(void* a);

#endif /* _NEWJOY_MATMAN_HPP_ */
