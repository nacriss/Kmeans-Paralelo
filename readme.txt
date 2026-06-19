# Paralelização do Algoritmo K-Means

## Autoras

* Ana Cristina Martins Silva
* Letícia Azevedo Cota Barbosa
* Lívia Alves Ferreira

---

# Descrição da Aplicação

Este projeto implementa o algoritmo de agrupamento **K-Means** em linguagem **C**, explorando diferentes estratégias de paralelização utilizando:

* OpenMP para CPU Multicore;
* OpenMP para GPU;
* CUDA para GPU.

Foi utilizada a base de dados **US_Accidents_March23.csv**, contendo aproximadamente **7,7 milhões de registros de acidentes de trânsito nos Estados Unidos**.

As coordenadas geográficas (**latitude** e **longitude**) foram utilizadas como atributos para a formação dos agrupamentos.

O valor adotado para o número de clusters foi:

```text
K = 50
```

A escolha foi motivada pela intenção de representar regiões geográficas distintas, tornando inadequada uma quantidade muito pequena de agrupamentos.

---

# Implementação Sequencial

| Versão     | Tempo        |
| ---------- | ------------ |
| Sequencial | **28,470 s** |

---

# OpenMP para Multicore (CPU)

## Estratégia de Paralelização

Após a análise do algoritmo, verificou-se que o principal gargalo computacional encontra-se nos **Steps 3 e 4**, responsáveis por:

1. Encontrar o centróide mais próximo de cada observação;
2. Reatribuir os pontos aos clusters.

Foi utilizada a diretiva:

```c
#pragma omp parallel for reduction(+:changed) private(t)
```

### Justificativa

* `private(t)` cria uma cópia privada da variável temporária para cada thread;
* `reduction(+:changed)` realiza a soma segura da variável que contabiliza as alterações de cluster.

Durante os testes também foi avaliada a paralelização do **Step 2 (recalcular centróides)** utilizando mecanismos de sincronização como:

* `critical`
* `atomic`

Entretanto, os resultados apresentaram elevada contenção entre threads, reduzindo significativamente o desempenho. Por esse motivo, optou-se por manter essa etapa sequencial.

---

## Resultados Obtidos

| Threads | Tempo (s) |
| ------- | --------- |
| 1       | 28,61     |
| 2       | 15,17     |
| 4       | 8,41      |
| 8       | 8,53      |
| 16      | 8,22      |
| 32      | 8,21      |

### Speedup

| Threads | Speedup |
| ------- | ------- |
| 1       | 1,00    |
| 2       | 1,89    |
| 4       | 3,40    |
| 8       | 3,35    |
| 16      | 3,48    |
| 32      | 3,49    |

### Análise

Observa-se que o ganho de desempenho cresce rapidamente até 4 threads.

A partir desse ponto, o speedup tende a estabilizar, indicando a presença de regiões sequenciais no algoritmo, conforme previsto pela Lei de Amdahl.

O melhor resultado foi obtido com **16 e 32 threads**, alcançando um speedup próximo de **3,5×**.

---

# OpenMP para GPU

## Estratégia de Implementação

A versão GPU foi desenvolvida utilizando **OpenMP Target Offloading**, transferindo para a GPU o principal gargalo computacional do algoritmo.

A função auxiliar responsável pelo cálculo da distância foi marcada com:

```c
#pragma omp declare target
```

permitindo sua execução diretamente na GPU.

A região paralelizada utiliza:

```c
#pragma omp target teams distribute parallel for reduction(+:changed) private(t)
```

### Papel das diretivas

* `target`: envia a execução para a GPU;
* `teams`: cria equipes de threads;
* `distribute`: distribui o trabalho entre as equipes;
* `parallel for`: distribui as iterações entre as threads;
* `private(t)`: evita condições de corrida;
* `reduction(+:changed)`: soma de forma segura as alterações realizadas.

---

## Gerenciamento de Dados

Os dados permanecem alocados na GPU através de:

```c
#pragma omp target data \
map(tofrom: observations[0:size]) \
map(alloc: clusters[0:k])
```

A atualização dos centróides é realizada por:

```c
#pragma omp target update to(clusters[0:k])
```

reduzindo o volume de dados transferidos pelo barramento PCIe.

---

## Resultados Obtidos

| Configuração | Tempo (s) |
| ------------ | --------- |
| GPU OpenMP   | **9,492** |

### Speedup

| Comparação                     | Speedup   |
| ------------------------------ | --------- |
| GPU OpenMP vs Sequencial       | **3,00×** |
| GPU OpenMP vs CPU (32 threads) | **0,86×** |

### Análise

Embora a GPU tenha apresentado ganho expressivo em relação à execução sequencial, o desempenho foi limitado por:

* Transferências de dados entre CPU e GPU;
* Necessidade de atualização dos centróides na CPU;
* Presença de partes sequenciais no algoritmo.

---

# CUDA para GPU

## Estratégia de Implementação

A versão CUDA foi desenvolvida utilizando kernels executados diretamente pela GPU.

O principal kernel implementado foi:

```c
assignClustersKernel
```

responsável por encontrar o centróide mais próximo para cada observação.

Cada thread processa exatamente uma observação.

---

## Configuração da GPU

Foram utilizados:

```c
256 threads por bloco
```

e:

```c
(size + threadsPerBlock - 1) / threadsPerBlock
```

para o cálculo dinâmico do número de blocos.

Para evitar acessos inválidos à memória:

```c
if (i >= size) return;
```

---

## Gerenciamento de Memória

Foi utilizada **Memória Unificada**:

```c
cudaMallocManaged()
```

permitindo um espaço de memória compartilhado entre CPU e GPU.

---

## Controle de Concorrência

Para evitar condições de corrida na variável `changed` foi utilizada a operação:

```c
atomicAdd()
```

garantindo atualizações seguras por múltiplas threads.

---

## Resultados Obtidos

| Configuração | Tempo (s)  |
| ------------ | ---------- |
| GPU CUDA     | **15,709** |

### Speedup

| Comparação         | Speedup   |
| ------------------ | --------- |
| CUDA vs Sequencial | **1,81×** |

### Análise

O desempenho foi limitado principalmente por:

* Latência de comunicação CPU ↔ GPU;
* Atualização sequencial dos centróides;
* Custos de sincronização entre host e device.

Mesmo assim, a implementação CUDA demonstrou ganhos significativos em relação à execução sequencial.

---

# Conclusões

Os resultados demonstraram que a paralelização do algoritmo K-Means pode reduzir significativamente o tempo de execução para grandes volumes de dados.

Entre as abordagens avaliadas:

| Implementação | Melhor Tempo |
| ------------- | ------------ |
| Sequencial    | 28,470 s     |
| OpenMP CPU    | 8,21 s       |
| OpenMP GPU    | 9,492 s      |
| CUDA GPU      | 15,709 s     |

A versão **OpenMP para CPU multicore** apresentou o melhor desempenho geral para o ambiente utilizado, atingindo speedup próximo de **3,5×** em relação à versão sequencial.
