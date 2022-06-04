/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans.h   (an OpenMP version)                            */
/*   Description:  header file for a simple k-means clustering program       */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

int omp_kmeans(int, float**, int, int, int, float, int*, float**);
int seq_kmeans(float**, int, int, int, float, int*, float**);

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*, int);

int read_n_objects(int, char*, int, int, float**);

int check_repeated_clusters(int, int, float**);

double  wtime(void);

extern int _debug;

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
double euclid_dist_2(double coord1,   /* [numdims] */
                    double coord2)   /* [numdims] */
{
    return (coord1-coord2) * (coord1-coord2);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         double  object,      /* [numCoords] */
                         double *clusters)    /* [numClusters] */
{
    int   index, i;
    double dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

#endif
