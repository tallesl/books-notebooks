import tensorflow as tf
from tensorflow import keras

all_text = 'Charmander is a bipedal, reptilian Pokémon with a primarily orange body and blue eyes. A fire burns at the tip of this Pokémon\'s slender tail, which has blazed there since Charmander\'s birth.'

window_size = 10
batch_size = 3

# converts a string to a
vectorizer = keras.layers.TextVectorization(standardize='lower_and_strip_punctuation', split='character')
vectorizer.adapt(all_text)

vocabulary = vectorizer.get_vocabulary()
vocabulary_size = len(vocabulary)

def vector_to_char(vector):
    return vocabulary[vector.numpy().item()]

def vector_to_string(vector):
    return ''.join(vocabulary[i] for i in vector)

def batch_to_string(batch):
    return ', '.join([f'"{vector_to_string(vector)}"' for vector in batch])

def split_input_and_target(windows):
    input = windows[:, :-1] # excludes the last token of the window, window size - 1 vector
    target = windows[:, 1:]  # excludes the first token of the window, windows size - 1 vector
    return (input, target)


all_text_vectorized = vectorizer(all_text)


print()
print('from_tensor_slices -> creates the dataset from a vector of indexes')
dataset = tf.data.Dataset.from_tensor_slices(all_text_vectorized)
for i, vector in dataset.enumerate().take(3):
    print(f'#{i}: {vector}')
    print(f'#{i}: \'{vector_to_char(vector)}\'')
    print()

print()
print('----------')
print()


print('window -> sets a window of 10, but the are returned as dataset itselfs')
dataset = dataset.window(window_size, shift=1, drop_remainder=True)
for i, vector in dataset.enumerate().take(3):
    print(f'#{i}: {vector}')

print()
print('----------')
print()


print('flat_map -> flattens the inner dataset created by window and returns as a vector')
dataset = dataset.flat_map(lambda window: window.batch(window_size))
for i, vector in dataset.enumerate().take(3):
    print(f'#{i}: {vector}')
    print(f'#{i}: \'{vector_to_string(vector)}\'')
    print()

print()
print('----------')
print()

print('print -> shuffles elements from now on, by 10000 blocks')
#dataset = dataset.shuffle(10000)

print()
print('----------')
print()

print()
print('batch -> returns batches of 3 windows (which itself is a vector of 10 characters')
dataset = dataset.batch(batch_size)
for i, batch in dataset.enumerate().take(3):
    print(f'#{i}: {batch}')
    print(f'#{i} \'{batch_to_string(batch)}\'')

print()
print('----------')
print()

print('map split_input_and_target -> for each vector of 10, makes two vectors of 9 as input and target, the first vector doesnt contain the last character and the latter vector doesnt contain the first character')
dataset = dataset.map(split_input_and_target)
for i, input_target in dataset.enumerate().take(3):
    print(f'#{i}: {input_target}')
    input, target = input_target
    print(f'#{i} input: \'{batch_to_string(input)}\'')
    print(f'#{i} target: \'{batch_to_string(target)}\'')
    print()

print()
print('----------')
print()

from pprint import pformat
#https://www.youtube.com/watch?v=BecEHOVmx9o
print('map one-hot')
print('because printing one-hot encoding is too verbose, well print only the first one-hot encoded character (first one-hot array of the first input sequence array of the first batch, which is the letter "c"')
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=vocabulary_size), Y_batch))
first_element = dataset.take(1).get_single_element()[0][0][0].numpy().tolist()
print(f'One-hot \'c\': {first_element}')
print(f'Zipped with vocabulary: {pformat(dict(zip(vocabulary, first_element)))}')

print()
print('----------')
print()

print('prefetch -> pre-process the next element for performance, autotune to left TF decide how many')
dataset = dataset.prefetch(tf.data.AUTOTUNE)



print()
print('----------')
print()

print(dataset.cardinality().numpy())
