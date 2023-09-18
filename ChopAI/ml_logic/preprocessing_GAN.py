############################################################################
# Imports
############################################################################

# Traditional imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Image and music specific imports
from imageio import imwrite
from music21 import converter, instrument, note, chord, converter
from PIL import Image, ImageOps

############################################################################
# Converting one midi file into an image
############################################################################

# Intermediary function
def extractNote(element):
    return int(element.pitch.ps)

#################################

# Intermediary function
def extractDuration(element):
    return element.duration.quarterLength

#################################

# Intermediary function
def get_notes(notes_to_parse):

    """
    Get all the notes and chords from the midi files into a dictionary containing:
        - Start: unit time at which the note starts playing
        - Pitch: pitch of the note
        - Duration: number of time units the note is played for
    """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))

        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

#################################

def midi2image(midi_path, output_folder_path, max_repetitions = float("inf"), resolution = 0.25, lowerBoundNote = 21, upperBoundNote = 127, maxSongLength = 100):

    """
    1) Transform a midi file into a set of images:
        - Each image has a size of 106 (all notes between lowerBound and upperBound) x 106 time units (maxSongLength)
        - One time unit corresponds to 0.25 (resolution) beat from the original music
    2) Store images into the corresponding sub-folder (identified by music piece name) of the 'output_folder_path' folder

    --> midi_path: path to midi file (e.g., '../../data_test/Input_midi/ballade2.mid')
    --> output_folder_path: path to folder where we wish to save created images (e.g., '../../data_test/Input_image/')

    """

    output_folder = f"{output_folder_path}{midi_path.split('/')[-1].replace('.mid', '')}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            notes_data = get_notes(notes_to_parse)
            if len(notes_data["start"]) == 0:
                continue

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = notes_data
                i+=1
            else:
                data[instrument_i.partName] = notes_data

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0"] = get_notes(notes_to_parse)

    for instrument_name, values in data.items():

        pitches = values["pitch"]
        durs = values["dur"]
        starts = values["start"]

        index = 0
        while index < max_repetitions:
            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))


            for dur, start, pitch in zip(durs, starts, pitches):
                dur = int(dur/resolution)
                start = int(start/resolution)

                if not start > index*(maxSongLength+1) or not dur+start < index*maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0 and j - index*maxSongLength < maxSongLength:
                            matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255

            if matrix.any(): # If matrix contains no notes (only zeros) don't save it
                output_filename = os.path.join(output_folder, midi_path.split('/')[-1].replace(".mid",f"_{instrument_name}_{index}.png"))
                imwrite(output_filename,matrix.astype(np.uint8))
                index += 1
            else:
                break

############################################################################
# Converting a dataset of midi files into a dataset of images
############################################################################

def get_clean_midi_data_as_images(midi_folder_path, output_folder_path, image_height = 106, image_length = 106):

    """
    Iterate on all midi files from the 'midi_folder_path' folder to:
        - Keep music pieces with one piano only
        - Transform the midi file into images
        - Store all corresponding images into a 'music_piece' subfolder of the 'output_folder_path'

    --> midi_folder_path: path to the folder where all midi files are stored
    --> output_folder_path: path to the folder where we wish to save created images
    """
    # Storing all midi files into a 'files_raw' list
    files_raw = [file for file in os.listdir(midi_folder_path)]

    # Storing all midi files with only one piano in a 'files' list
    files = []
    for file in files_raw:
        try:
            mid = converter.parse(f'{midi_folder_path}/{file}')
            file_instruments = instrument.partitionByInstrument(mid)
            if len(file_instruments)==1:
                files.append(file)
        except:
            pass

    # Iterating on all files from 'files' list to create images
    for file in files:
        file_path = f"{midi_folder_path}/{file}"
        midi2image(file_path, output_folder_path)

############################################################################
# Cleaning dataset of images
############################################################################

def clean_images(input_folder_path, output_folder_path, height_image = 106, length_image = 106):
    """
    Iterate on all images created in the 'input_path' folder:
        - Resize images to height_image x length_image
        - Transform them into pure black and white images
        - Save them in a 'music piece' subfolder of the 'output_path' folder

    --> Input path: path to folder with input images (e.g., '../../data_test/Input_image')
    --> Output path: path to folder where we wish to save output reshaped images (e.g., '../../data_test/Input_image_cleaned')
    """

    for music in os.listdir(input_folder_path):

        output_folder = f'{output_folder_path}/{music}' # Creating one sub_folder for each music piece in the 'output_path' folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for image in os.listdir(f"{input_folder_path}/{music}"):
            image_path = f'{input_folder_path}/{music}/{image}'
            image_read = Image.open(image_path) # Reading each image
            new_image = image_read.resize((106,106)) # Resizing each image
            new_image = new_image.convert("1") # Convert each image to pure black and white
            new_image.save(f'{output_folder}/{image}') # Saving each image

############################################################################
# Creating an array of images to be used as dataset in Python
############################################################################

def get_pixels_array(input_folder_path):
    """
    Generate an array containing all images from 'input_path' folder in array format

    --> input_path = path of the folder containing clean images (e.g., '../../data_image_cleaned')
    """

    pixels = []
    for music in os.listdir(input_folder_path):
        for image in os.listdir(f"{input_folder_path}/{music}"):
            image_path = f'{input_folder_path}/{music}/{image}'
            image_read = Image.open(image_path) # Reading each image
            pixels_image = np.array(image_read.getdata()).astype('float32') # Store all pixel values in an array, each i_th-sequence contains the values of pixels in a i_th-row
            pixels_image = pixels_image / 255.0 # All the values are 0 (black) and white (255). Normalize pixel values to be between 0 and 1
            pixels.append(pixels_image.reshape(106, 106,1)) # Reshape pixels to be a matrix

    pixels = np.array(pixels)

    return pixels

#################################

# Custom function for dataset visualization
def show_image_from_pixels(pixels_matrix, image_number):
    """
    Given a pixel matrix representing a dataset, get representation of one image (number image_number)
    """
    plt.imshow(np.squeeze(pixels_matrix[image_number, :, :, :]))
    plt.show()
