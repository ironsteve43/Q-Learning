import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

# Definição dos índices das ações
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def read_input_file(file_name):
    with open(file_name, 'r') as f:
        # Lê a primeira linha com os parâmetros i, a, g, r, e
        params = f.readline().split()
        i, a, g, r = map(float, params[:4])
        e = float(params[4]) if len(params) >= 5 else 0.0

        # Lê a segunda linha com o tamanho N da matriz
        N = int(f.readline())

        # Lê o restante do arquivo como a matriz de recompensas
        data = np.loadtxt(f)

    return i, a, g, r, e, N, data

def initialize_q_table(N, data):
    Q = np.random.uniform(-0.5, 0.5, size=(N, N, 4))  # Inicializa com valores aleatórios entre -0.5 e 0.5

    # Define os valores terminais com as recompensas
    Q[data == 7] = 1
    Q[data == 4] = -1

    # Define os valores para os estados obstáculos como 0
    Q[data == -1] = 0

    return Q


def q_learning(i, a, g, r, e, N, data):
    Q = initialize_q_table(N, data)

    # Define os valores terminais com as recompensas
    Q[data == 7] = 1
    Q[data == 4] = -1

    # Índices dos estados com recompensa +1 e -1
    positive_reward_state = np.argwhere(data == 7)[0]
    negative_reward_state = np.argwhere(data == 4)[0]
    initial_state = np.argwhere(data == 10)[0]
    data_list = []  # Lista para armazenar os estados do tabuleiro em cada episódio

    # Loop principal do Q-Learning
    for _ in range(int(i)):
        # Posição inicial do agente (estado com label 10)
        current_state = np.argwhere(data == 10)[0]
        data_list.append(data.copy())  
        # Loop para cada episódio
        while True:
            # Escolher uma ação usando a política e-greedy, se "e" for fornecido
            if e > 0 and np.random.rand() < e:
                # Ação aleatória
                action = np.random.choice([UP, DOWN, LEFT, RIGHT])
            else:
                # Ação com maior valor Q
                action = np.argmax(Q[current_state[0], current_state[1]])

              # Determinar ação final com base na probabilidade de escorregar (10% de chance)
            if random.random() < 0.2:

            # Determina a nova posição após escorregar
                sliding_directions = [(action + 1) % 4, (action - 1) % 4]
                sliding_action = random.choice(sliding_directions)
                action = sliding_action
                 
            # Executar a ação no ambiente e obter a próxima posição
            next_state = current_state.copy()
            if action == 0:
                next_state[0] -= 1
            elif action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] += 1
            elif action == 3:
                next_state[1] -= 1

            # Verificar se a próxima posição é um estado válido (dentro dos limites da matriz e não é um obstáculo)
            if 0 <= next_state[0] < N and 0 <= next_state[1] < N and data[next_state[0], next_state[1]] != -1:
                # Obter a recompensa do próximo estado
                if data[next_state[0], next_state[1]] == 4:
                    reward = -1
                elif data[next_state[0], next_state[1]] == 7:
                    reward = 1
                else:
                    reward = r
                
                # Atualizar o valor de Q usando a equação do Q-Learning
                best_next_action = np.argmax(Q[next_state[0], next_state[1]])
                Q[current_state[0], current_state[1], action] = (1 - a) * Q[current_state[0], current_state[1], action] + a * (reward + g * Q[next_state[0], next_state[1], best_next_action])
                
                # Atualizar a posição do agente no tabuleiro
                if data[next_state[0], next_state[1]] != 4 and data[next_state[0], next_state[1]] != 7:
                    data[next_state[0], next_state[1]] = 10
                
                data[current_state[0], current_state[1]] = 0
                data_list.append(data.copy()) 

                # Atualizar o estado atual do agente para o próximo estado
                current_state = next_state.copy()
               
            else:
                # A ação não é válida (leva a um estado inválido), o agente continua na posição atual
                next_state = current_state.copy()

            # Verificar se o episódio terminou (alcançou um estado terminal)
            if current_state[0] == positive_reward_state[0] and current_state[1] == positive_reward_state[1]:
                data_list.append(data.copy()) 
                data[current_state[0], current_state[1]] = 0
                data[positive_reward_state[0], positive_reward_state[1]] = 10 #Agente entra no estado
                data_list.append(data.copy())
                data[initial_state[0], initial_state[1]] = 10 #Agente volta para a posição inicial
                data[positive_reward_state[0], positive_reward_state[1]] = 7
                current_state = initial_state
                break
            if current_state[0] == negative_reward_state[0] and current_state[1] == negative_reward_state[1]:
                data_list.append(data.copy()) 
                data[current_state[0], current_state[1]] = 0
                data[negative_reward_state[0], negative_reward_state[1]] = 10 #Agente entra no estado
                data_list.append(data.copy())
                data[initial_state[0], initial_state[1]] = 10 #Agente volta para a posição inicial
                data[negative_reward_state[0], negative_reward_state[1]] = 4
                current_state = initial_state
                break
            
            data_list.append(data.copy())        

    return Q, data_list

def generate_gif(data_list, N, output_file):
    fig = plt.figure()

    def init():
        sns.heatmap(np.zeros((N, N)), square=True, cbar=False)

    def animate(i):
        data = data_list[i]
        sns.heatmap(data, square=True, cbar=False)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data_list), repeat=False)

    output_folder = "saida"
    output_path = os.path.join(output_folder, output_file + ".gif")
    
    pillow_writer = animation.PillowWriter(fps=7)
    anim.save(output_path, writer=pillow_writer)
     
    
def generate_heatmap_image(Q, data, output_file):
    actions = ['c', 'd', 'b', 'e', 'n']
    max_q_values = np.max(Q, axis=2)
    best_actions = np.argmax(Q, axis=2)
    print(best_actions)
    
    labels = np.array([[actions[best_actions[i, j]] if (data[i, j] != -1 and data[i, j] != 4 and data[i, j] != 7)  else 'n' for j in range(len(data[0]))] for i in range(len(data[0]))])
    print(labels)
    
    data_with_rewards = data.copy()
    data_with_rewards[data_with_rewards != 0] = np.nan

    plt.figure()
    sns.heatmap(max_q_values, cbar=True, square=True, annot=labels, fmt='')
    
    output_folder = "saida"
    output_path = os.path.join(output_folder, output_file + "_acoes.png")
    
    plt.savefig(output_path)
   
     
def main(input_file, output_file):
    inputs = read_input_file("arquivodeentrada.txt") #Leitura da entrada
    Q_values, data_list = q_learning(*inputs)
    for acao in range(4): #Impressão dos Q_values para cada ação
        print(f"Q-values para a ação {acao}:")
        print(np.round(Q_values[:, :, acao], 3))
        print("\n")

    generate_heatmap_image(Q_values, inputs[6], output_file) #Geração do heatmap
    print(np.average(np.max(Q_values, axis=2))) #Recompensa média
    generate_gif(data_list, inputs[5], "saidateste") #Geração do gif


if __name__ == "__main__":
    input_file = "arquivodeentrada.txt"
    output_file = "saidateste"
    main(input_file, output_file)          