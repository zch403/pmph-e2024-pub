#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    for(int i=0; i<tot_size; i++) { flags_d[i] = 0; }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    for(int i=0; i<mat_rows; i++) { flags_d[mat_shp_sc_d[i]] = 1; }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    for(int i=0; i<tot_size; i++) { tmp_pairs[i] = mat_vals[i]*vct[mat_inds[i]] }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    for(int i=0; i<mat_rows; i++) { res_vct_d[i] = tmp_scan[mat_shp_sc_d[i]]; }
}

#endif // SPMV_MUL_KERNELS
