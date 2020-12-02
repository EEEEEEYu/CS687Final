import pandas as pd
import os


def split_dataset(path,count=10):
    with open(path,'r') as read_file:
        # Read total count 1000000
        total_episode_count = int(read_file.readline())

        # Split dataset into 10 groups. Each group has two files storing index and episodes
        for index in range(count):
            episode_finished = 0
            index_file_path = "data/small_dataset_index_" + str(index)+".csv"
            episode_file_path = "data/small_dataset_" + str(index)+".csv"

            # Remove existing files
            if os.path.exists(index_file_path):
                os.remove(index_file_path)
            if os.path.exists(episode_file_path):
                os.remove(episode_file_path)

            # Open file
            index_write_file = open(index_file_path,'a')
            episode_write_file = open(episode_file_path, 'a')
            while episode_finished < int(total_episode_count/count):
                episode_length = int(read_file.readline())

                # Write index
                index_write_file.write(str(episode_length)+'\n')

                # Write episode
                for row in range(episode_length):
                    episode_write_file.write(read_file.readline())
                episode_finished += 1

            # Close files
            index_write_file.close()
            episode_write_file.close()
            print("file "+str(index)+" finished!")
    print("finished splitting files!")


split_dataset('data/data.csv')