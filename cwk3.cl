// Kernel for matrix transposition.
__kernel
void matrixTranspose( __global float *inputMatrix, __private int nRows, __private int nCols, __global float *outputMatrix)
{
	int
		i = get_global_id(0),
        j = get_global_id(1);

    for(i=0; i<nRows; i++){
        for(j=0; j<nCols; j++){
            outputMatrix[j*nRows+i] = inputMatrix[i*nRows+j];
        }
    }    

}