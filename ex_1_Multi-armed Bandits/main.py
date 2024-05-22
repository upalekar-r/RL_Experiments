from env import BanditEnv
from agent import BanditAgent, EpsilonGreedy, UCB
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


def q4(k: int, num_samples: int):
    env = BanditEnv(k=k)
    env.reset()
    rewards = np.zeros((num_samples, k))

    for j in range(k):
        for i in range(num_samples):
            reward = env.step(j)  # Directly store the returned reward
            rewards[i, j] = reward

    # Create the violins
    plt.violinplot(rewards, positions=np.arange(1, k+1), showmeans=True, showextrema=True)

    # Add labels and title
    plt.xlabel('Actions')
    plt.ylabel('Rewards')
    plt.title('Violin Plot of the Reward Distribution per Arm')
    plt.xticks(np.arange(1, k+1), np.arange(1, k+1))  # Set x-ticks to match the number of arms
    plt.grid(True)
    plt.show()


def q6(k: int, trials: int, steps: int):
    
    env = BanditEnv(k=k)
    agent1 = EpsilonGreedy(k,0,0)
    agent2 = EpsilonGreedy(k,0,0.01)
    agent3 = EpsilonGreedy(k,0,0.1)
    q_max = []
   
    Array_reward1=[]
    Array_reward2=[]
    Array_reward3=[]

    R_optimal_array1=[]
    R_optimal_array2=[]
    R_optimal_array3=[]

    # Iterate Through the Trials
    for t in trange(trials, desc="Trials"):
        # Reinitialize the Environment and Agents at the Beginning of Each Trial
        env.reset()
        q_max.append(env.means.max())
    
        agent1.reset()
        agent2.reset()
        agent3.reset()

        R1=[] 
        R2=[]
        R3=[] 
        R_optimal1=[]  
        R_optimal2=[] 
        R_optimal3=[] 
       
        for step in range(steps):
            action1 = agent1.choose_action()
            action2 = agent2.choose_action()
            action3 = agent3.choose_action()
            reward1= env.step(action1)
            reward2= env.step(action2)
            reward3= env.step(action3)
            agent1.update(action1,reward1)
            agent2.update(action2,reward2)
            agent3.update(action3,reward3)
            R1.append(reward1)
            R2.append(reward2)
            R3.append(reward3)
            if action1 == np.argmax(env.means):
                R_optimal1.append(1)
            else:
                R_optimal1.append(0)

            if action2 == np.argmax(env.means):
                R_optimal2.append(1)
            else:
                R_optimal2.append(0)

            if action3 == np.argmax(env.means):
                R_optimal3.append(1)
            else:
                R_optimal3.append(0)
        
        R_optimal_array1.append(R_optimal1)
        R_optimal_array2.append(R_optimal2)
        R_optimal_array3.append(R_optimal3)
       
        Array_reward1.append(R1)
        Array_reward2.append(R2)
        Array_reward3.append(R3)
    R_final1 = np.vstack(Array_reward1)
    R_final2 = np.vstack(Array_reward2)
    R_final3 = np.vstack(Array_reward3)


    Reward_average1 = np.mean(R_final1, axis = 0)
    Reward_average2 = np.mean(R_final2, axis = 0)
    Reward_average3 = np.mean(R_final3, axis = 0)
    average_q_max = np.average(q_max)
    average_r_optimal1 = np.mean(R_optimal_array1, axis = 0)
    average_r_optimal2 = np.mean(R_optimal_array2, axis = 0)
    average_r_optimal3 = np.mean(R_optimal_array3, axis = 0)

    per_optimal_r1 = list((x*100 for x in average_r_optimal1 ))
    per_optimal_r2 = list((x*100 for x in average_r_optimal2 ))
    per_optimal_r3 = list((x*100 for x in average_r_optimal3 ))

    average_R_std1 = np.std(R_final1, axis = 0)
    average_R_std2 = np.std(R_final2, axis = 0)
    average_R_std3 = np.std(R_final3, axis = 0)
    average_q_std = average_q_max/np.sqrt(trials)

    band1 = average_R_std1/np.sqrt(trials)
    band2 = average_R_std2/np.sqrt(trials)
    band3 = average_R_std3/np.sqrt(trials)
    band4 = average_q_max/np.sqrt(trials) 

    agent1_upper = (Reward_average1 + 1.96*band1)
    agent1_lower = (Reward_average1 - 1.96*band1)
    agent2_upper = (Reward_average2 + 1.96*band2)
    agent2_lower = (Reward_average2 - 1.96*band2)
    agent3_upper = (Reward_average3 + 1.96*band3)
    agent3_lower = (Reward_average3 - 1.96*band3)
    upper_limit_higher = (average_q_max + 1.96*band4)
    upper_limit_lower = (average_q_max - 1.96*band4)

    

    fig, (ax1, ax2) = plt.subplots(2, 1)
    x_axis = np.arange(steps)

    ax1.plot(x_axis,Reward_average1, label = 'e = 0', color = 'green')
    ax1.plot(x_axis,Reward_average2, label = 'e = 0.01', color = 'red')
    ax1.plot(x_axis,Reward_average3, label = 'e = 0.1', color = 'blue')
    ax1.plot(x_axis,[average_q_max]*len(x_axis), label = 'upper limit', color = 'black')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
   


    ax2.plot(per_optimal_r1, label = 'e = 0', color = 'green')
    ax2.plot(per_optimal_r2, label = 'e = 0.01', color = 'red')
    ax2.plot(per_optimal_r3, label = 'e = 0.1', color = 'blue')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% optimalimal Action')

    ax1.fill_between(x_axis,agent1_lower,agent1_upper,alpha=0.2,color='green')
    ax1.fill_between(x_axis,agent2_lower,agent2_upper,alpha=0.2,color='red')
    ax1.fill_between(x_axis,agent3_lower,agent3_upper,alpha=0.2,color='blue')
    ax1.fill_between(x_axis,upper_limit_lower,upper_limit_higher,alpha=0.2,color='black')

    ax2.legend()
    
    plt.show()

    

def q7(k: int, trials: int, steps: int):
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    agent1 = EpsilonGreedy(k,0,0,0.1)
    agent2 = EpsilonGreedy(k,5,0,0.1)
    agent3 = EpsilonGreedy(k,0,0.1,0.1)
    agent4 = EpsilonGreedy(k,5,0.1,0.1)
    agent5 = UCB(k,0,2,0.1)
    q_max = []
    Array_reward1=[]
    Array_reward2=[]
    Array_reward3=[]
    Array_reward4=[]
    Array_reward5=[]

    R_optimal_array1=[]
    R_optimal_array2=[]
    R_optimal_array3=[]
    R_optimal_array4=[]
    R_optimal_array5=[]

    action_list=[]
    # Iterate Through the Trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        q_max.append(env.means.max())
    
        agent1.reset()
        agent2.reset()
        agent3.reset()
        agent4.reset()
        agent5.reset()

        R1=[] 
        R2=[]
        R3=[] 
        R4=[] 
        R5=[] 
        R_optimal1=[]  
        R_optimal2=[] 
        R_optimal3=[] 
        R_optimal4=[]
        R_optimal5=[]
        action_list2=[]

        for step in range(steps):
            action1 = agent1.choose_action()
            action2 = agent2.choose_action()
            action3 = agent3.choose_action()
            action4 = agent4.choose_action()
            action5 = agent5.choose_action()
            if step in range(12):
                action_list2.append(action2)
            reward1= env.step(action1)
            reward2= env.step(action2)
            reward3= env.step(action3)
            reward4= env.step(action4)
            reward5= env.step(action5)

            agent1.update(action1,reward1)
            agent2.update(action2,reward2)
            agent3.update(action3,reward3)
            agent4.update(action4,reward4)
            agent5.update(action5,reward5)

            R1.append(reward1)
            R2.append(reward2)
            R3.append(reward3)
            R4.append(reward4)
            R5.append(reward5)

            if action1 == np.argmax(env.means):
                R_optimal1.append(1)
            else:
                R_optimal1.append(0)

            if action2 == np.argmax(env.means):
                R_optimal2.append(1)
            else:
                R_optimal2.append(0)

            if action3 == np.argmax(env.means):
                R_optimal3.append(1)
            else:
                R_optimal3.append(0)

            if action4 == np.argmax(env.means):
                R_optimal4.append(1)
            else:
                R_optimal4.append(0)

            if action5 == np.argmax(env.means):
                R_optimal5.append(1)
            else:
                R_optimal5.append(0)
        R_optimal_array1.append(R_optimal1)
        R_optimal_array2.append(R_optimal2)
        R_optimal_array3.append(R_optimal3)
        R_optimal_array4.append(R_optimal4)
        R_optimal_array5.append(R_optimal5)

        Array_reward1.append(R1)
        Array_reward2.append(R2)
        Array_reward3.append(R3)
        Array_reward4.append(R4)
        Array_reward5.append(R5)
        action_list.append(action_list2)

    R_final1 = np.vstack(Array_reward1)
    R_final2 = np.vstack(Array_reward2)
    R_final3 = np.vstack(Array_reward3)
    R_final4 = np.vstack(Array_reward4)
    R_final5 = np.vstack(Array_reward5)
    
    Reward_average1 = np.mean(R_final1, axis = 0)
    Reward_average2 = np.mean(R_final2, axis = 0)
    Reward_average3 = np.mean(R_final3, axis = 0)
    Reward_average4 = np.mean(R_final4, axis = 0)
    Reward_average5 = np.mean(R_final5, axis = 0)
    average_q_max = np.average(q_max)
    
    average_r_optimal1 = np.mean(R_optimal_array1, axis = 0)
    average_r_optimal2 = np.mean(R_optimal_array2, axis = 0)
    
    average_r_optimal3 = np.mean(R_optimal_array3, axis = 0)
    average_r_optimal4 = np.mean(R_optimal_array4, axis = 0)
    average_r_optimal5 = np.mean(R_optimal_array5, axis = 0)

    per_optimal_r1 = list((x*100 for x in average_r_optimal1 ))
    per_optimal_r2 = list((x*100 for x in average_r_optimal2 ))
    per_optimal_r3 = list((x*100 for x in average_r_optimal3 ))
    per_optimal_r4 = list((x*100 for x in average_r_optimal4 ))
    per_optimal_r5 = list((x*100 for x in average_r_optimal5 ))

    average_R_std1 = np.std(R_final1, axis = 0)
    average_R_std2 = np.std(R_final2, axis = 0)
    average_R_std3 = np.std(R_final3, axis = 0)
    average_R_std4 = np.std(R_final4, axis = 0)
    average_R_std5 = np.std(R_final5, axis = 0)
    average_q_std = average_q_max/np.sqrt(trials)

    band1 = average_R_std1/np.sqrt(trials)
    band2 = average_R_std2/np.sqrt(trials)
    band3 = average_R_std3/np.sqrt(trials)
    band4 = average_R_std4/np.sqrt(trials)
    band5 = average_R_std5/np.sqrt(trials)
    band6 = average_q_max/np.sqrt(trials) 

    agent1_upper = (Reward_average1 + 1.96*band1)
    agent1_lower = (Reward_average1 - 1.96*band1)
    agent2_upper = (Reward_average2 + 1.96*band2)
    agent2_lower = (Reward_average2 - 1.96*band2)
    agent3_upper = (Reward_average3 + 1.96*band3)
    agent3_lower = (Reward_average3 - 1.96*band3)
    agent4_upper = (Reward_average4 + 1.96*band4)
    agent4_lower = (Reward_average4 - 1.96*band4)
    agent5_upper = (Reward_average5 + 1.96*band5)
    agent5_lower = (Reward_average5 - 1.96*band5)
    upper_limit_higher = (average_q_max + 1.96*band6)
    upper_limit_lower = (average_q_max - 1.96*band6)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    x_axis = np.arange(steps)

    ax1.plot(x_axis,Reward_average1, label = 'Q1=0,e=0', color = 'green')
    ax1.plot(x_axis,Reward_average2, label = 'Q1=5,e=0', color = 'red')
    ax1.plot(x_axis,Reward_average3, label = 'Q1=0,e=0.1', color = 'blue')
    ax1.plot(x_axis,Reward_average4, label = 'Q1=5,e=0.1', color = 'orange')
    ax1.plot(x_axis,Reward_average5, label = 'UCB c=2', color = 'brown')
    ax1.plot(x_axis,[average_q_max]*len(x_axis), label = 'upper limit', color = 'black')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.legend()


    ax2.plot(per_optimal_r1, label = 'Q1=0,e=0', color = 'green')
    ax2.plot(per_optimal_r2, label = 'Q1=5,e=0', color = 'red')
    ax2.plot(per_optimal_r3, label = 'Q1=0,e=0.1', color = 'blue')
    ax2.plot(per_optimal_r4, label = 'Q1=5,e=0.1', color = 'orange')
    ax2.plot(per_optimal_r5, label = 'UCB c=2', color = 'brown')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')

    ax1.fill_between(x_axis,agent1_lower,agent1_upper,alpha=0.2,color='green')
    ax1.fill_between(x_axis,agent2_lower,agent2_upper,alpha=0.2,color='red')
    ax1.fill_between(x_axis,agent3_lower,agent3_upper,alpha=0.2,color='blue')
    ax1.fill_between(x_axis,agent4_lower,agent4_upper,alpha=0.2,color='orange')
    ax1.fill_between(x_axis,agent5_lower,agent5_upper,alpha=0.2,color='brown')
    ax1.fill_between(x_axis,upper_limit_lower,upper_limit_higher,alpha=0.2,color='black')
    ax2.legend()
    plt.show()


def main():
    # TODO run code for all questions
    q4(10,2000)
    q6(10,2000,10000)
    q7(10,2000,10000)


if __name__ == "__main__":
    main()
