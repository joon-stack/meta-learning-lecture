import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class DataLoader(object):
    def __init__(self, data_type:str='train', n_way:int=5, n_support:int=5, n_query:int=5):
        self.data_type = data_type
        self.data = self.preprocess_data(tfds.load("omniglot", split=self.data_type, as_supervised=True))
        
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.task_list = None
    
        
    def preprocess_data(self, dataset):
        print(f"Preprocessing {self.data_type} Omniglot dataset")
        def preprocess(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28,28])
            return image, label

        data = {}
        for image, label in dataset.map(preprocess):
            image = image.numpy()
            label = str(label.numpy())
            if label not in data:
                data[label] = []
            data[label].append(image)
        print("Finished preprocessing")
        return data
        
    def generate_task_list(self, n_tasks:int=100, n_way:int=0, n_query:int=0, n_support:int=0):
        n_way = self.n_way if n_way == 0 else n_way
        n_query = self.n_query if n_query == 0 else n_query
        n_support = self.n_support if n_support == 0 else n_support
        task_list = list()
        ############### Your code here ###################
            # TODO: finish implementing this method.
            # Append n_tasks number of tasks to task_list
            # where each task is a dictionary in the form of
            # {label: [random sequence]}
            # Hint: the keys of self.data can be used as labels

        ##################################################
        
        for i in range(n_tasks):
            task_keys = np.random.permutation(list(self.data.keys()))
            task = {}
            for j in range(n_way):
                seq = np.random.permutation(n_query + n_support)
                key = task_keys[j]
                task[key] = seq
            task_list.append(task)
        
        self.task_list = task_list
        
    def delete_task_list(self):
        self.task_list = None
    
    def visualize_random_task(self):
        s, _ = self.data_generator()

        for img in s:
            fig, axs = plt.subplots(1, self.n_support, figsize=(self.n_support, self.n_way))
            if self.n_support > 1:
                for i in range(self.n_support):
                    axs[i].imshow(img[i], cmap='gray')
                    axs[i].axis('off')
            elif self.n_support == 1:
                axs.imshow(img[0], cmap='gray')
                axs.axis('off')

        plt.show()
    
    def data_generator(self, task_idx=0):       
        if self.task_list != None:
            # Deterministic task from predefined task space
            assert(task_idx >= 0)
            task = self.task_list[task_idx]

        else:
            self.generate_task_list(n_tasks=1)
            task = self.task_list[0]
            self.delete_task_list()
    
        
        support = np.zeros([self.n_way, self.n_support, 28, 28, 1], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 28, 28, 1], dtype=np.float32)

        ############### Your code here ###################
            # TODO: finish implementing this method.
            # Using a task generated by generate_task_list,
            # create a support and query dataset with shapes
            # (n_way, n_support/n_query, 28, 28, 1)

        ##################################################
        
        for i, key in enumerate(list(task.keys())):
            values = task[key]
            images = self.data[key]
            for j in range(self.n_support):
                idx = values[j]
                support[i][j] = tf.reshape(images[idx], (28, 28, 1))
            for j in range(self.n_query):
                idx = values[j + self.n_support]
                query[i][j] = tf.reshape(images[idx], (28, 28, 1))
        
        return support, query

    def random_data_generator(self):
        n_ways = [2, 3, 5, 6, 10, 15]
        n_shots = [15, 10, 6, 5, 3, 2]
        
        n_way = 0
        n_support = 0
        n_query = 0
        
        ############### Your code here ###################
            # TODO: Find numbers for n_way, n_support, n_query
            # where n_way * (n_support + n_query) == 30
        

        ##################################################
        
        idx = np.random.randint(1, len(n_ways))
        n_way = n_ways[idx]
        n_shot = n_shots[idx]
        # print("{} way {} shot task".format(n_way, n_shot))
        
        # Modify n_support and n_query
        n_support = np.random.randint(1, n_shot)
        n_query = n_shot - n_support
        # print("# support: {}, # query: {}".format(n_support, n_query))
        
        assert(n_support > 0 and n_query > 0)
    
        assert(n_way * (n_support + n_query) == 30)
        
        
        # Generate a random task
        self.generate_task_list(n_tasks=1, n_way=n_way, n_support=n_support, n_query=n_query)
        task = self.task_list[0]
        self.delete_task_list()
        
        support = np.zeros([n_way, n_support, 28, 28, 1], dtype=np.float32)
        query = np.zeros([n_way, n_query, 28, 28, 1], dtype=np.float32)
    
        ############### Your code here ###################
            # TODO: finish implementing this method.
            # create a support and query dataset with shapes
            # (n_way, n_support/n_query, 28, 28, 1)
            # (Same as in the data_generator method)

        ##################################################
        for i, key in enumerate(list(task.keys())):
            values = task[key]
            images = self.data[key]
            for j in range(n_support):
                idx = values[j]
                support[i][j] = tf.reshape(images[idx], (28, 28, 1))
            for j in range(n_query):
                idx = values[j + n_support]
                query[i][j] = tf.reshape(images[idx], (28, 28, 1))
        
        return support, query