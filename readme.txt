RELATÓRIO – PARALELIZAÇÃO DO ALGORITMO K-MEANS

AUTORAS

* Ana Cristina Martins Silva
* Letícia Azevedo Cota Barbosa
* Lívia Alves Ferreira

======================================================================

1. DESCRIÇÃO DA APLICAÇÃO
   ======================================================================

Este projeto implementa o algoritmo de agrupamento K-Means em linguagem C.

Utilizamos a base de dados US_Accidents_March23.csv, contendo aproximadamente
7,7 milhões de registros de acidentes de trânsito nos Estados Unidos. Foram
utilizadas as coordenadas geográficas (latitude e longitude) para formar os
agrupamentos.

O número de clusters escolhido para esse dataset foi K = 50, pois não faria
sentido uma quantidade muito pequena de clusters, já que a ideia é agrupar os
acidentes por estados.

======================================================================
2. IMPLEMENTAÇÃO SEQUENCIAL
===========================
Tempo obtido:

Versão: Sequencial
Tempo: 28,470 s

======================================================================
3. OPENMP PARA MULTICORE (CPU)
==============================

3.1 Estratégia de Paralelização

A paralelização foi realizada utilizando a biblioteca OpenMP.

Após análise do algoritmo, observou-se que a etapa mais adequada para
paralelização foi o processo de reatribuição dos pontos aos centróides
(Steps 3 e 4).

Foi utilizada a diretiva:

#pragma omp parallel for reduction(+:changed) private(t)

onde:

* private(t) evita condições de corrida sobre a variável temporária;
* reduction(+:changed) permite a soma segura da variável responsável por
  contabilizar mudanças de cluster.

Durante os testes verificamos que a paralelização do Step 2 (recalcular
centróides) fazia uma grande contenção entre as threads, reduzindo o
desempenho. Dessa forma, para não alterar a estrutura original do algoritmo
K-Means, optamos por manter essa etapa sequencial.

---

## 3.2 Resultados Obtidos

## Threads      Tempo (s)

1            28,61
2            15,17
4             8,41
8             8,53
16            8,22
32            8,21

---

## 3.3 Speedup

Considerando o tempo com uma thread como referência:

## Threads      Speedup

1            1,00
2            1,89
4            3,40
8            3,35
16           3,48
32           3,49

Observa-se que o melhor desempenho foi obtido com 16 e 32 threads,
apresentando speedup próximo de 3,5 vezes em relação à execução com apenas
uma thread.

Também foi observado que o ganho deixa de crescer significativamente após
4 threads, indicando a presença de partes sequenciais do algoritmo que
limitam a escalabilidade.

======================================================================
4. OPENMP PARA GPU
==================

4.1 Estratégia de Implementação

A paralelização do algoritmo K-Means na GPU foi desenvolvida utilizando o
modelo de execução por diretivas OpenMP Target Offloading.

O objetivo principal foi mover o gargalo computacional do algoritmo para a
GPU, gerenciando o ciclo de vida dos dados para minimizar o custo de
transferência pelo barramento PCIe.

A região paralelizada compreende os Steps 3 e 4, responsáveis pela
reatribuição dos pontos aos clusters.

A função auxiliar calculateNearst foi marcada com:

#pragma omp declare target

permitindo a geração de uma versão da função para execução na GPU.

Para o laço de reatribuição foi utilizada a diretiva:

#pragma omp target teams distribute parallel for reduction(+:changed) private(t)

onde:

* teams cria equipes de execução na GPU;
* distribute distribui as iterações entre as equipes;
* parallel for distribui o trabalho entre as threads;
* private(t) garante uma cópia privada da variável temporária;
* reduction(+:changed) realiza a soma segura das alterações de cluster.

Para o gerenciamento dos dados foi utilizada a diretiva:

#pragma omp target data map(tofrom: observations[0:size]) 
map(alloc: clusters[0:k])

mantendo o dataset alocado na memória da GPU durante toda a execução.

A sincronização dos centróides foi realizada através de:

#pragma omp target update to(clusters[0:k])

permitindo atualizar apenas o vetor de clusters a cada iteração.

---

## 4.2 Resultados Obtidos

## Configuração         Tempo (s)

GPU OpenMP            9,492

---

## 4.3 Speedup

Considerando como referência a versão sequencial pura, executada com uma
thread, a versão OpenMP para GPU alcançou speedup aproximado de 3,00 vezes.

Quando comparada ao melhor resultado obtido pela CPU multicore (32 threads),
a GPU apresentou speedup relativo de aproximadamente 0,86 vezes.

Esse comportamento evidencia o impacto das regiões sequenciais do algoritmo
e do custo de transferência de dados entre CPU e GPU.

Apesar dessas limitações, a implementação demonstrou que o modelo de
offloading é capaz de acelerar significativamente o processamento de grandes
volumes de dados.

======================================================================
5. CUDA PARA GPU
================

5.1 Estratégia de Implementação

A paralelização utilizando CUDA foi desenvolvida explorando diretamente os
recursos da GPU através de kernels.

O principal kernel implementado foi:

assignClustersKernel

responsável pela etapa de associação de cada observação ao centróide mais
próximo.

Cada thread da GPU processa uma observação do dataset.

Foi utilizada uma configuração composta por:

* 256 threads por bloco;
* grid calculado dinamicamente através da expressão:

(size + threadsPerBlock - 1) / threadsPerBlock

Além disso, foi utilizada a condição:

if (i >= size) return;

para evitar acessos inválidos à memória.

---

## 5.2 Gerenciamento de Memória

Foi utilizada Memória Unificada através da função:

cudaMallocManaged()

Essa estratégia reduz a necessidade de cópias explícitas entre CPU e GPU,
simplificando o gerenciamento de memória.

---

## 5.3 Controle de Concorrência

Para evitar condições de corrida na variável "changed", foi utilizada a
operação:

atomicAdd()

Essa operação garante que múltiplas threads possam atualizar a variável de
forma segura sem perda de dados.

---

## 5.4 Resultados Obtidos

## Configuração         Tempo (s)

GPU CUDA             15,709

---

## 5.5 Speedup

Comparando a execução CUDA com a versão sequencial, foi obtido um speedup
aproximado de 1,81 vezes.

O desempenho foi limitado principalmente pela transferência de dados entre
CPU e GPU e pela permanência da etapa de atualização dos centróides na CPU.

Mesmo assim, os resultados demonstram que a utilização da GPU proporciona
ganhos significativos para a etapa mais custosa do algoritmo.