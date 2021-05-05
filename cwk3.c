//
// Starting point for the OpenCL coursework for COMP3221 Parallel Computation.
//
// Once compiled, execute with the number of rows and columns for the matrix, e.g.
//
// ./cwk3 16 8
//
// This will display the matrix, followed by another matrix that has not been transposed
// correctly. You need to implement OpenCL code so that the transpose is correct.
//
// For this exercise, both the number of rows and columns must be a power of 2,
// i.e. one of 1, 2, 4, 8, 16, 32, ...
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 3 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// - getCmdLineArgs(): Gets the command line arguments and checks they are valid.
// - displayMatrix() : Displays the matrix, or just the top-left corner if it is too large.
// - fillMatrix()    : Fills the matrix with random values.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
    //
    // Parse command line arguments and check they are valid. Handled by a routine in the helper file.
    //
    int nRows, nCols;
    getCmdLineArgs( argc, argv, &nRows, &nCols );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the matrix.
    float 
        *hostMatrix = (float*) malloc( nRows*nCols*sizeof(float) ),
        *transposedMatrix = (float*) malloc( nRows*nCols*sizeof(float) );
    
    // Fill the matrix with random values, and display.
    fillMatrix( hostMatrix, nRows, nCols );
    printf( "Original matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nRows, nCols );

    //Creating corresponding matrices and variables to pass to the device.
    cl_mem device_matrix = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nRows*nCols*sizeof(float),hostMatrix, &status );
    cl_int device_nRows = nRows;
    cl_int device_nCols = nCols;
    cl_mem device_transposedMatrix = clCreateBuffer( context, CL_MEM_WRITE_ONLY ,  nRows*nCols*sizeof(float), NULL, &status);
    
    //
    // Transpose the matrix on the GPU.
    //

    //Building Kernel code.
    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "matrixTranspose", context, device);

    //Setting kernel arguments.
    status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &device_matrix); 
    status = clSetKernelArg( kernel, 1, sizeof(cl_int), &device_nRows); 
    status = clSetKernelArg( kernel, 2, sizeof(cl_int), &device_nCols); 
    status = clSetKernelArg( kernel, 3, sizeof(cl_mem), &device_transposedMatrix);
    
    //Setting up global problem size and work group size.
    size_t	indexSpaceSize[1], workGroupSize[1];
    indexSpaceSize[0] = nCols*nRows;
	workGroupSize [0] = 2;	

    //Put the kernel onto command queue.
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
    if (status != CL_SUCCESS){
        printf("Kernel enqueuing failed: ERROR %d.\n", status );
        return EXIT_FAILURE;
    }


    //Retrieve the result from device to host + error handling.
    status = clEnqueueReadBuffer(queue, device_transposedMatrix, CL_TRUE, 0, nCols*nRows*sizeof(float), transposedMatrix, 0, NULL, NULL);
    if( status != CL_SUCCESS )
	{
		printf( "Could not copy device data to host: Error %d.\n", status );
		return EXIT_FAILURE;
	}

    //Display the final result. The initial statement has been altered to output the matrix containing the transposed
    //version of hostMatrix - transposedMatrix; any extra allocated memory has been freed.
    printf( "Transposed matrix (only top-left shown if too large):\n" );
    displayMatrix( transposedMatrix, nCols, nRows);


    //
    // Release all resources.
    //
    clReleaseKernel(kernel);
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );
    clReleaseMemObject( device_matrix);
    clReleaseMemObject( device_transposedMatrix);

    free(hostMatrix);
    free(transposedMatrix);
    return EXIT_SUCCESS;
}

