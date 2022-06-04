/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         mpi_kmeans.c  (MPI version)                               */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "KmeansMPI.h"


/*----< mpi_kmeans() >-------------------------------------------------------*/
int mpi_kmeans(double     *objects,     /* in: [numObjs][numCoords] */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *&membership,  /* out: [numObjs] */
               double    *&clusters)    /* out: [numClusters][numCoords] */
               // MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    double  *newClusters;    /* [numClusters][numCoords] where numCords==1*/
    double  *origClusters;    /* [numClusters][numCoords] where numCords==1*/
    int _debug = 0;

    if (_debug) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    // newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    newClusters    = new double[numClusters];
    origClusters    = new double[numClusters];
    assert(newClusters != NULL);
    for (i=0; i<numClusters; i++) {
        newClusters[i] = 0.0;
        origClusters[i] = clusters[i];
    }

    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (_debug) printf("%2d: numObjs=%d total_numObjs=%d numClusters=%d \n",rank,numObjs,total_numObjs,numClusters);

    do {
        double curT = MPI_Wtime();
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, objects[i], clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            newClusters[index] += objects[i];
        }

        /* sum all data objects in newClusters */
        MPI_Allreduce(newClusters, clusters, numClusters,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            if (clusterSize[i] > 1)
                clusters[i] /= clusterSize[i];
            newClusters[i] = 0.0;  /* set back to 0 */
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        delta = delta_tmp / total_numObjs;

        if (_debug) {
            double maxTime;
            curT = MPI_Wtime() - curT;
            MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            // if (rank == 0)
                // printf("%2d: loop=%d time=%f sec\n",rank,loop,curT);
        }
    } while (delta > threshold && loop++ < 500);
    if (_debug && rank == 0)
        printf("%2d: delta=%f threshold=%f loop=%d\n",rank,delta,threshold,loop);

    free(newClusters);
    free(newClusterSize);
    free(clusterSize);

    return 1;
}

