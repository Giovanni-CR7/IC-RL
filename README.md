# Projeto

Este repositório implementa agentes RL (PPO / opcionalmente RecurrentPPO) para o problema de Coverage Path Planning (CPP) em grids 2D com obstáculos. O agente observa uma janela local e um vetor global (posição normalizada, razão de cobertura, passos restantes e direção-objetivo calculada por BFS) e aprende a maximizar a cobertura com penalidades por revisitas e colisões.

## Conteúdo

- **Código principal:** [train_cpp.py](train_cpp.py) — treino, teste, execução e curriculum.
- **Comparação:** [compare.py](compare.py) — avalia o modelo RL contra algoritmos clássicos e salva CSVs.
- **Ambiente:** [gymnasium_env/grid_world_cpp_v2.py](gymnasium_env/grid_world_cpp_v2.py) — definição do ambiente, observação, reward e render.
- **Dependências:** [requirements.txt](requirements.txt)
- **Resultados:** arquivos CSV gerados por avaliações (ex.: [resultados_5x5.csv](resultados_5x5.csv), [resultados_7x7.csv](resultados_7x7.csv), [resultados_10x10.csv](resultados_10x10.csv), [resultados_15x15.csv](resultados_15x15.csv), [resultados_20x20.csv](resultados_20x20.csv)).

## Resumo técnico

- **Espaço de observação:** Dict com `local` (janela local flatten lv×lv) e `global` (lista: `ax/(size-1)`, `ay/(size-1)`, `coverage_ratio`, `steps_remaining`, e quando ativado `dx, dy` — direção-objetivo calculada por BFS). Veja [gymnasium_env/grid_world_cpp_v2.py](gymnasium_env/grid_world_cpp_v2.py).
- **Espaço de ações:** Discreto(4) — movimentos: direita, cima, esquerda, baixo.
- **Recompensa:** passo base `-0.1`; colisão `-0.5`; nova célula `+1.0`; revisita `-0.3`; cobertura completa `+10.0`; truncamento por limite de passos `-5.0`.
- **Política / Algoritmo:** PPO com `MultiInputPolicy` (suporta observações Dict). Opcionalmente `RecurrentPPO` (LSTM) via `sb3-contrib` para memória intra-episódio.

## Curriculum (estágios)

Os estágios do curriculum são definidos em [train_cpp.py](train_cpp.py) e, por padrão, seguem 5 → 7 → 10 → 15 → 20 com timesteps por estágio:

| Estágio | Grid | max_steps | timesteps |
|---:|:---:|:---:|---:|
| 1 | 5×5  | 50  | 200_000 |
| 2 | 7×7  | 98  | 300_000 |
| 3 | 10×10| 200 | 500_000 |
| 4 | 15×15| 450 | 1_000_000 |
| 5 | 20×20| 800 | 2_000_000 |

As quantidades de obstáculos por estágio são calculadas no código como `max(1, round(0.12 * size * size))` quando o curriculum é executado.

> **Nota:** a tabela acima descreve os valores **padrão** do comando `curriculum`. Os modelos avaliados na seção de Resultados abaixo **não** foram treinados via `curriculum` — cada um foi treinado individualmente com `train`, usando obstáculos e timesteps menores e específicos por grid (ver tabela de Resultados). Use a tabela do Curriculum como referência do comportamento padrão desse comando, não como a receita exata dos modelos avaliados.

## Como treinar

- Crie e ative um ambiente virtual (ex.: PowerShell): `python -m venv env; .\env\Scripts\Activate.ps1`.
- Instale dependências: `pip install -r requirements.txt`.
- Treino único (exemplo 10×10, PPO-MLP): `python train_cpp.py train --dim 10 --obstacles 8 --max-steps 200 --timesteps 500000`.
- Treino recorrente (RecurrentPPO, requer sb3-contrib): `python train_cpp.py train --dim 10 --obstacles 8 --max-steps 200 --timesteps 500000 --recurrent`.
- Curriculum (treina 5→7→10→15→20): `python train_cpp.py curriculum` — use `--resume data/model.zip` para retomar.

## Como avaliar / comparar

- Avaliar um `.zip` salvo (determinístico): `python train_cpp.py test --dim 10 --obstacles 8 --max-steps 200 --model data/meu_modelo.zip --episodes 100`.
- Comparar RL vs. clássicos e salvar CSV: `python compare.py --model data/meu_modelo.zip --dim 10 --obstacles 8 --max-steps 200 --episodes 100 --csv resultados_10x10.csv`.

## Resultados (avaliações)

As tabelas a seguir foram extraídas dos CSVs gerados por `compare.py` no repositório. Cada linha refere-se ao agente RL treinado (PPO) nas mesmas sementes/instâncias usadas para os algoritmos clássicos.

| Grid | Obstáculos | max_steps | Cobertura completa (RL) | Cobertura média (RL) | Desvio (RL) | Passos médios (RL) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5×5  | 2  | 50  | 100.0% | 100.00% | 0.00% | 23.9 |
| 7×7  | 4  | 98  | 99.0%  | 99.93%  | 0.66% | 49.7 |
| 10×10| 8  | 200 | 99.0%  | 99.79%  | 2.05% | 114.7 |
| 15×15| 18 | 450 | 96.0%  | 99.56%  | 2.77% | 277.3 |
| 20×20| 32 | 800 | 90.0%  | 99.37%  | 2.51% | 507.6 |

Observação: os algoritmos clássicos (BFS, Boustrophedon, Spanning Tree, Frontier) costumam alcançar 100% de cobertura nas mesmas instâncias (veja os CSVs para detalhes). Use [compare.py](compare.py) para reproduzir comparações exatas.

## Arquivos de interesse e convenções

- **Modelos salvos:** diretório `data/` — nomes seguem o padrão gerado por `make_run_paths`/curriculum (ex.: `data/ppo_cpp_mlp_10_8_200_20260527_120257.zip` ou `data/ppo_cpp_lstm_curr_15x15_obs18_steps450_2026...zip`).
- **Logs:** diretório `log/` — subpastas por execução com CSVs e TensorBoard.

## Scripts utilitários

- **`eval_save_10x10.py`** — avalia um modelo salvo em `num_episodes` episódios, rodando **duas vezes** (uma com política estocástica, outra determinística) e salvando cada episódio (flag determinístico, cobertura, passos) em CSV. Uso: `python eval_save_10x10.py <modelo> --size 10 --obstacles 12 --max_steps 200 --episodes 100 --out results/eval_10x10.csv`.
- **`fine_tune_10x10.py`** — carrega um modelo `.zip` já treinado e continua o treinamento (fine-tuning) por mais passos numa configuração de grid específica (padrão 10×10, 12 obstáculos), salvando o modelo resultante com timestamp e logs de TensorBoard. Uso: `python fine_tune_10x10.py <modelo> [timesteps]` (padrão 500.000 timesteps).

## Recomendações práticas

- Para desenvolvimento rápido, treine primeiro em 5×5 ou 7×7 com poucos timesteps para validar mudanças.
- Ao avaliar com `train_cpp.py test`, a trava anti-deadlock fica ativada por padrão; desative com `--no-watchdog` se quiser medir a política pura. Essa flag não existe em `compare.py`, que já avalia sempre com política determinística pura (sem watchdog).
- Para replicar resultados numéricos exatos, fixe a semente (parâmetro `--seed` em `compare.py`) e execute `--episodes` suficientes (100 é usado nos CSVs aqui).