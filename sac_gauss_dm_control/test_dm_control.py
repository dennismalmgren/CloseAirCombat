#dm control has max of 1 reward per time step (?)
#this results in a return prediction of max 100.

from torchrl.envs import DMControlEnv

def main():
    env = DMControlEnv("reacher", "easy")
    td = env.rand_step()
    print(td)
    

if __name__=="__main__":
    main()