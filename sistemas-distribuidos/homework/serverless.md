## Cloud
    - Iaas, Paas, Saas são padrões pela NIST
    - começou devido a virtualização (ESX) e a internet
    - containers e separação de grupos de recursos, provido pelo BSD (jails) e Linux (cgroups), também ajudaram

## Alguns tipos de Cloud

### Multi-Cloud
    - usar multiplos serviços de diferentes providers ao mesmo tempo
    - mais dificil de gerenciar

### Hybrid Cloud
    - usar nuvem privada e publica ao mesmo tempo
    - também é uma tarefa complicada devida a certas implicações

### Federated Cloud
    - Trazer varias clouds para o mesmo guarda-chuva
    - reduz o lock in de providers

### microcloud
    - redução de latência
    - redução do footprint no meio ambiente
    - uso de hardware low-cost (raspberry pi, odroid)
    - redução da comunicação entre user e datacenter
    - maneira mais decentralizada
    
### cloudlet
    - parecido com microcloud
    - usado no contexto de mobile
    - usado para diminuir latência e QoS de aplicações mobile
    
### Ad Hoc Cloud
    - usar aparelhos comuns de usuários comuns para aproveitar o processamento que não é totalmente aproveitado
    - problemas com usuários maliciosos
        - semelhante à uma botnet
    - uso de mobile de usuários para uma rede grande pode diminuir a vida útil de aparalhos
        - usuários precisam estar cientes e ser voluntários
    - P2P por exemplo com bittorrent
    - pode ser visto como `social cloud computing`
    - ideia de `fog computing`, da qual dispositivos básicos são usados aos montes para integrar uma rede de aumentar a capacidade de processamento e diminuição de latência geral
    - também há a ideia de `edge computing`, da qual é utilizado redes móveis para reduzir trafego

### Heterogeneous Cloud
    - pode ser no contexto de multiclouds
    - pode ser à nivel de arquitetura, tendo diversas arquiteturas disponíveis
    
### Apenas interessante (SDN):
    - Software Defined Networking (SDN)
    - maneira de isolar o hardware dos componentes que controlam o trafego dedados
    - permite a codificação dos componentes que controlam a rede
    - arquitetura dinâmica, usada para evoluir junto com o volume de dados gerados hoje
    - Há também SDC (Software defined computing) do qual tem o objetivo de reconfigurar e adaptar recursos físicos
    - usados para QoS
    - permite ambientes mais heterogêneos e a granularidade das configurações

### 5 Caracteristicas de cloud
    1. sob demanda
    2. acesso amplo a rede (de qualquer local)
    3. aglomerado de recursos distribuidos
    4. rápida eslaticidade (facil de escalar)
    5. Coleta de métricas


## Serverless
    - No modelo serverful, o consumidor paga por toda a máquina
        - Nem todos os clientes utilizam toda a máquina
    - Serveless surgiu como uma maneira do cliente pagar apenas por sua aplicação, e não pela máquina
    - ainda há servidor mesmo sendo serverless
    - parte de uma máquina é alugada
    - período menor de tempo
    - custo é referente a execução de determinada porção de código (custo granular)
    - abstração de complexidade
        - não há necessidade de saber sobre infraestrutura ou outras configurações
    - provedor se encarrega de provisionar, gerenciar e escalar servidores máquinas e containers
    - não há uma definição formal
    - criado para reduzir a complexidade de manter uma máquina na nuvem
    - similar to PaaS
    - não tem estado devido ao curto tempo de vida
    - segue a ideia de microserviços
        - decomposição do software em partes pequenas que podem rodar de forma independente
    - pode ser visto como uma maneira de executar e hospedar microserviços
    - quando um evento é disparado, o provedor sobe a instancia do container/vm que possui a função serverless e após a execução esses recursos são liberados e reciclados
    - armazenamento é separado das instâncias
    - o desenvolvedor pré-define os eventos que invocarão a função
    - desenvolvedor foca apenas na lógica
    - última parte depois de anos de evolução em containers e virtualização
    - transição de `bare metal` para `bare code`
    - começou na decada de 1968, com o CICS da IBM. Uma ideia bem primitiva de executar, programas de usuário aparti de transações (eventos)
        - a evolução também se deu devido ao RPC e CGI

## Exemplos Serverless
    - AWS Lambda
    - Google Serverless Computing
    - Azure Functions

## exemplos de uso
    - app de camera, do qual utiliza Faas para funções de edição
    - aplicação IOT que recolhe informações e funções serverless são usadas para processar os dados

## Serverless é usado em:
    - Machine Learning
    - IOT
    - Análises
    - Big Data

## Tipos:

### Faas (Function as a Service)
    - execução baseada em eventos
    - tempo de execução limitada
    - ocioso até um evento acontecer
    - isolados
    - containers
    - operantes até a pedida da retirada do ar pelo criador
    - contém: controlador, fonte de eventos e instâncias das funções
        - controlador: gerencia o desenvolvimento das funções, gerencia da escalabilidade, monitoramento das instâncias, controle de funções e fonte de eventos
        - fontes de eventos: acionam ou transmite eventos para as instâncias
        - instâncias: ambientes de execução para as funções

### Baas (Backend as a Service)
    - serviços terceiros sobre demanda
    - abstração passada para o frontend
    - exemplos: 2FA, Login, DB, File Store

## Vantagens:
    - escalonamento ágil (tanto vertical como horizontal)
    - sem despesas por ociosidade
    - sem custo por escalabilidade
    - redução do custo operacional
    - fast time to market
    - uso eficiente de recursos

## Desvantagens:
    - Desempenho (tempos de inicialização são custosos)
    - Latência de inicialização (cold start) (quando recursos precisam ser criados isso leva mais tempo do que um warm start)
    - lock in do provedor (ficar preso à um provedor)
    - segurança
    - precificação (dependendo do que for usado pode sair mais custoso que uma máquina virtual completa)
    - Sistema legado (mais dificil de traduzir para serverless)
    - tempo de execução (tempo é restringido pelo provedor, então deve-se levar em consideração para o armazenamento de informações e execução em si)   
    - Limitações de Recursos

## Futuro:
    - Hardware accelerators (GPUs, FPGAs)
    - multi-cloud
    - stateful
    - melhor predição de preços
    - migração para serverless

## Bad practices:
    - Asynchronous calls
        - aumentam a complexidade do sistema
        - deve ser implementada de forma sincrona ou usar pub-sub
        - pode ser usado como gatilho para outras tarefas longas
    - Chamar outras funções
        - dificulta o debugging
        - diminui a isolação
        - junte as funções quando possível
    - código compartilhado entre funções
        - pode quebrar funções existentes
        - atingir limites de tamanho
        - piorar o tempo de inicialização
        - escreva funções desacopladas
    - usar muitas bibliotecas
        - use apenas o necessário
        - em AWS Lambda (se o tempo de inicialização não for problema) carregue as bibliotecas no inicio
    - usar muitas tecnologias diferentes
        - aumenta a complexidade de manutenção
        - limitar o numero de tecnologias usadas
    - muitas funções
        - diminui a mantenabilidade
        - agrupar funções em microserviços
        - considerar se vale a pena criar uma nova função ou alterar uma existente

## Referências:

- [Desvendando as nuvens: uma análise comparativa de desempenho de aplicações serverless e containers em provedores de computação em nuvem](https://repositorio.unesp.br/entities/publication/fa5f4662-d479-4fa2-9d7b-8676914d0da3)
- [Rise of the Planet of Serverless Computing: A Systematic Review](https://dl.acm.org/doi/full/10.1145/3579643)
- [Serverless: What it Is, What to Do and What Not to Do](https://ieeexplore.ieee.org/abstract/document/9095731)
- [Serverless is More: From PaaS to Present Cloud Computing](https://ieeexplore.ieee.org/abstract/document/8481652)
- [Next generation cloud computing: New trends and research directions](https://www.sciencedirect.com/science/article/abs/pii/S0167739X17302224)
