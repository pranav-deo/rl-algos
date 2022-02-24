import os
from tracemalloc import start
import imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import gym    
import cv2

def label_with_episode_number(frame, episode_num, step_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 40)

    drawer.text((im.size[0]/20,im.size[1]/18), f'Epoch: {episode_num}', font=fnt, fill=text_color)
    drawer.text((im.size[0]*12/20,im.size[1]/18), f'Step: {step_num}', font=fnt, fill=text_color)

    return im

def combine_gifs(agent_name, env_name, start_epoch, end_epoch, interval, num_rows, num_columns):
    gifs_name = [f'./{agent_name}/{env_name}/{ii}/result.gif' for ii in range(start_epoch,end_epoch+interval,interval)]
    gifs = [imageio.get_reader(file) for file in gifs_name]

    #If they don't have the same number of frame take the shorter
    number_of_frames = np.max([gif.get_length() for gif in gifs]) 

    #Create writer object
    new_gif = imageio.get_writer(f'./{agent_name}/{env_name}/combined_results.gif')


    for frame_number in range(number_of_frames-1):
        imgs = []
        for gif in gifs:
            if gif.get_length() > frame_number:
                img = gif.get_data(frame_number)
            else:
                img = gif.get_data(gif.get_length()-1)

            width, height = int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)
            imgs.append(cv2.resize(img, (width, height), cv2.INTER_LANCZOS4))

        new_images = [np.hstack(tuple(imgs[i:i+num_columns])) for i in range(0, num_columns*num_rows, num_columns)]
        new_image = np.vstack(new_images)
        new_gif.append_data(new_image)

    for gif in gifs:
        gif.close()
    new_gif.close()

def plot_from_saved_metrics(algo, env):
    actor_losses = np.load(f'./{algo}/{env}/actor_losses.npy')
    critic_losses = np.load(f'./{algo}/{env}/critic_losses.npy')
    rewards = np.load(f'./{algo}/{env}/rewards.npy')

    plt.figure()
    plt.plot(np.arange(1, len(actor_losses)+1), actor_losses, label='Actor Losses')
    plt.plot(np.arange(1, len(critic_losses)+1), critic_losses, label='Critic Losses')
    plt.legend()
    plt.xlabel('Number of Epochs')
    plt.savefig(f'{algo}/{env}/actor_critic_losses.png', bbox_inches='tight')
    plt.show()
    plt.figure()
    plt.plot(np.arange(1, len(rewards)+1), rewards, label='reward')
    plt.legend()
    plt.xlabel('Number of Epochs')
    plt.savefig(f'{algo}/{env}/rewards.png', bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    env = gym.make('CartPole-v1')
    frames = []
    for i in range(5):
        state = env.reset()        
        for t in range(500):
            action = env.action_space.sample()

            frame = env.render(mode='rgb_array')
            frames.append(label_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)
