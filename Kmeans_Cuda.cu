/**
 * @file k_means_clustering.c
 * @brief K Means Clustering Algorithm implemented
 * @details
 * This file has K Means algorithm implemmented
 * It prints test output in eps format
 *
 * Note:
 * Though the code for clustering works for all the
 * 2D data points and can be extended for any size vector
 * by making the required changes, but note that
 * the output method i.e. printEPS is only good for
 * polar data points i.e. in a circle and both test
 * use the same.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad)
 */

#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */

typedef struct observation
{
    double x;  /**< abscissa of 2D data point */
    double y;  /**< ordinate of 2D data point */
    int group; /**< the group no in which this observation would go */
} observation;

typedef struct cluster
{
    double x;     /**< abscissa centroid of this cluster */
    double y;     /**< ordinate of centroid of this cluster */
    size_t count; /**< count of observations present in this cluster */
} cluster;


/*
int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}
*/
//função para substituir o gargalo
__global__ void assignClustersKernel(observation* obs, cluster* clusters, int size, int k, int* changed)
{
    //descobrindo o id global da thread a ser processada
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //para evitar que threads extras acessem memória que não existe
    if (i >= size) return;

    double minD = 1e30; 
    int bestCluster = -1;
    //calculo da distância permanece (ponto i até k clusters)
    for (int c = 0; c < k; c++)
    {
        double dist = (clusters[c].x - obs[i].x) * (clusters[c].x - obs[i].x) +
                      (clusters[c].y - obs[i].y) * (clusters[c].y - obs[i].y);
        if (dist < minD)
        {
            minD = dist;
            bestCluster = c;
        }
    }

    if (obs[i].group != bestCluster)
    {
        obs[i].group = bestCluster;
        atomicAdd(changed, 1); //evita que múltiplas threads atualizem a variável changed ao mesmo tempo
    }
}

void calculateCentroid(observation observations[], size_t size, cluster* centroid) {

    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = NULL;
    if (k <= 1)
    {
        //clusters = (cluster*)malloc(sizeof(cluster));
        cudaMallocManaged(&clusters, sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if (k < size)
    {
        //clusters = malloc(sizeof(cluster) * k);
        cudaMallocManaged(&clusters, sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        for (size_t j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        //size_t changed = 0;
        int* d_changed;
        cudaMallocManaged(&d_changed, sizeof(int)); //é compartilhada tbm
        size_t minAcceptedError = size /10000;  // Do until 99.99 percent points are in correct cluster
        int t = 0;

        // Configuração da malha da GPU (Blocos e Threads)
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        do
        {
            *d_changed = 0; //inicializa a variável de mudança para 0 antes de cada iteração
            /* Initialize clusters */
            //ingual
            for (int i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* STEP 2*/
            for (size_t j = 0; j < size; j++)
            {
                t = observations[j].group;
                clusters[t].x += observations[j].x;
                clusters[t].y += observations[j].y;
                clusters[t].count++;
            }
            for (int i = 0; i < k; i++)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }

            /*
            /* STEP 3 and 4 
            aqui chama o gargalo que substituimos
            changed = 0;  
            for (size_t j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }
            */
            //chama o kernel para atribuir os clusters e contar as mudanças
            assignClustersKernel<<<blocksPerGrid, threadsPerBlock>>>(observations, clusters, size, k, d_changed);
            cudaDeviceSynchronize(); //espera o kernel terminar para acessar a variável de mudança
        } while (*d_changed > minAcceptedError);  // Keep on grouping until we have
                                               // got almost best clustering
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        cudaMallocManaged(&clusters, sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        for (int j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
        cudaFree(clusters); //libera a memória alocada para clusters, já que não é mais necessária
    }
    return clusters;
}


void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    int i;
    size_t j;
    double kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x)
        {
            max_x = pts[j].x;
        }
        if (min_x > pts[j].x)
        {
            min_x = pts[j].x;
        }
        if (max_y < pts[j].y)
        {
            max_y = pts[j].y;
        }
        if (min_y > pts[j].y)
        {
            min_y = pts[j].y;
        }
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    };
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10,
           H + 10);
    printf(
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");
    for (int i = 0; i < k; i++)
    {
        printf("%g %g %g setrgbcolor\n", *(colors + 3 * i),
               *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i)
            {
                continue;
            }
            printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                   (pts[j].y - cy) * scale + H / 2);
        }
        printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
               (cent[i].y - cy) * scale + H / 2);
    }
    printf("\n%%%%EOF");

    // free accquired memory
    free(colors);
}

static void test()
{
    size_t size = 100000L;
    //observation* observations = (observation*)malloc(sizeof(observation) * size);
    observation* observations;
    cudaMallocManaged(&observations, sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 5;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    cudaFree(observations);
    cudaFree(clusters);
}

void test2()
{
    size_t size = 1000000L;
    //observation* observations = (observation*)malloc(sizeof(observation) * size);
    observation* observations;
    cudaMallocManaged(&observations, sizeof(observation) * size);
    double maxRadius = 20.00;
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 11;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    //printEPS(observations, size, clusters, k);
    // Free the accquired memory
    cudaFree(observations);
    cudaFree(clusters);
}

observation *loadDataset(const char *filename, size_t *size)
{
    FILE *file = fopen(filename, "r");

    if (!file)
    {
        printf("Erro ao abrir %s\n", filename);
        return NULL;
    }

    char line[8192];
    fgets(line, sizeof(line), file); // Pula a linha de cabeçalho

    size_t capacity = 1000000;

    observation *observations = (observation *)malloc(capacity * sizeof(observation));
    *size = 0;

    while (fgets(line, sizeof(line), file))
    {
        double latitude = 0;
        double longitude = 0;
        int column = 0;

        char *token = strtok(line, ",");
        while (token != NULL)
        {
            if (column == 5)
                latitude = atof(token);
            if (column == 6)
                longitude = atof(token);

            token = strtok(NULL, ",");
            column++;
        }

        if (*size >= capacity)
        {
            capacity *= 2;
            observations = (observation *)realloc(observations, capacity * sizeof(observation));
        }

        observations[*size].x = longitude;
        observations[*size].y = latitude;
        observations[*size].group = 0;

        (*size)++;
    }

    fclose(file);
    return observations;
}


void testAccidentsDataset()
{
    size_t size;

    // 1. Carrega os dados na CPU
    observation *host_observations = loadDataset("US_Accidents_March23.csv", &size);

    if (!host_observations)
        return;

    printf("Registros carregados: %zu\n", size);

    observation *observations;
    cudaMallocManaged(&observations, size * sizeof(observation));
    memcpy(observations, host_observations, size * sizeof(observation));
    free(host_observations); 

    int k = 50;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Iniciando agrupamento com %d clusters na GPU...\n", k);
    cudaEventRecord(start);

    cluster *clusters = kMeans(observations, size, k);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tempo total: %.3f segundos\n", milliseconds / 1000.0);

    cudaFree(clusters);
    cudaFree(observations);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    srand(42);
    testAccidentsDataset();
    return 0;
}