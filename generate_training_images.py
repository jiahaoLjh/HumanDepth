#!/usr/bin/env python
# coding: utf-8


import cv2
import os


subjects = [1, 5, 6, 7, 8, 9, 11]
actions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
subactions = [1, 2]

resolutions = {
    (1, 1): (1002, 1000),
    (1, 2): (1000, 1000),
    (1, 3): (1000, 1000),
    (1, 4): (1002, 1000),
    (5, 1): (1002, 1000),
    (5, 2): (1000, 1000),
    (5, 3): (1000, 1000),
    (5, 4): (1002, 1000),
    (6, 1): (1002, 1000),
    (6, 2): (1000, 1000),
    (6, 3): (1000, 1000),
    (6, 4): (1002, 1000),
    (7, 1): (1002, 1000),
    (7, 2): (1002, 1000),
    (7, 3): (1002, 1000),
    (7, 4): (1002, 1000),
    (8, 1): (1002, 1000),
    (8, 2): (1000, 1000),
    (8, 3): (1000, 1000),
    (8, 4): (1002, 1000),
    (9, 1): (1002, 1000),
    (9, 2): (1000, 1000),
    (9, 3): (1000, 1000),
    (9, 4): (1002, 1000),
    (11, 1): (1002, 1000),
    (11, 2): (1000, 1000),
    (11, 3): (1000, 1000),
    (11, 4): (1002, 1000),
}

camera_names = {
    1: "54138969",
    2: "55011271",
    3: "58860488",
    4: "60457274",
}
action_names = {
    2: "Directions",
    3: "Discussion",
    4: "Eating",
    5: "Greeting",
    6: "Phoning",
    7: "Posing",
    8: "Purchases",
    9: "Sitting",
    10: "SittingDown",
    11: "Smoking",
    12: "Photo",
    13: "Waiting",
    14: "Walking",
    15: "WalkDog",
    16: "WalkTogether",
}
subaction_names = {
    (2, 1): ["Directions 1",
            "Directions 1",
            "Directions 1",
            "Directions 1",
            "Directions 1",
            "Directions 1",
            "Directions 1"],
    (2, 2): ["Directions",
            "Directions 2",
            "Directions",
            "Directions",
            "Directions",
            "Directions",
            "Directions"],
    (3, 1): ["Discussion 1",
            "Discussion 2",
            "Discussion 1",
            "Discussion 1",
            "Discussion 1",
            "Discussion 1",
            "Discussion 1"],
    (3, 2): ["Discussion",
            "Discussion 3",
            "Discussion",
            "Discussion",
            "Discussion",
            "Discussion 2",
            "Discussion 2"],
    (4, 1): ["Eating 2",
            "Eating 1",
            "Eating 1",
            "Eating 1",
            "Eating 1",
            "Eating 1",
            "Eating 1"],
    (4, 2): ["Eating",
            "Eating",
            "Eating 2",
            "Eating",
            "Eating",
            "Eating",
            "Eating"],
    (5, 1): ["Greeting 1",
            "Greeting 1",
            "Greeting 1",
            "Greeting 1",
            "Greeting 1",
            "Greeting 1",
            "Greeting 2"],
    (5, 2): ["Greeting",
            "Greeting 2",
            "Greeting",
            "Greeting",
            "Greeting",
            "Greeting",
            "Greeting"],
    (6, 1): ["Phoning 1",
            "Phoning 1",
            "Phoning 1",
            "Phoning 2",
            "Phoning 1",
            "Phoning 1",
            "Phoning 3"],
    (6, 2): ["Phoning",
            "Phoning",
            "Phoning",
            "Phoning",
            "Phoning",
            "Phoning",
            "Phoning 2"],
    (7, 1): ["Posing 1",
            "Posing 1",
            "Posing 2",
            "Posing 1",
            "Posing 1",
            "Posing 1",
            "Posing 1"],
    (7, 2): ["Posing",
            "Posing",
            "Posing",
            "Posing",
            "Posing",
            "Posing",
            "Posing"],
    (8, 1): ["Purchases 1",
            "Purchases 1",
            "Purchases 1",
            "Purchases 1",
            "Purchases 1",
            "Purchases 1",
            "Purchases 1"],
    (8, 2): ["Purchases",
            "Purchases",
            "Purchases",
            "Purchases",
            "Purchases",
            "Purchases",
            "Purchases"],
    (9, 1): ["Sitting 1",
            "Sitting 1",
            "Sitting 1",
            "Sitting 1",
            "Sitting 1",
            "Sitting 1",
            "Sitting 1"],
    (9, 2): ["Sitting 2",
            "Sitting",
            "Sitting 2",
            "Sitting",
            "Sitting",
            "Sitting",
            "Sitting"],
    (10, 1): ["SittingDown 2",
             "SittingDown",
             "SittingDown 1",
             "SittingDown",
             "SittingDown",
             "SittingDown",
             "SittingDown"],
    (10, 2): ["SittingDown",
             "SittingDown 1",
             "SittingDown",
             "SittingDown 1",
             "SittingDown 1",
             "SittingDown 1",
             "SittingDown 1"],
    (11, 1): ["Smoking 1",
             "Smoking 1",
             "Smoking 1",
             "Smoking 1",
             "Smoking 1",
             "Smoking 1",
             "Smoking 2"],
    (11, 2): ["Smoking",
             "Smoking",
             "Smoking",
             "Smoking",
             "Smoking",
             "Smoking",
             "Smoking"],
    (12, 1): ["Photo 1", # "TakingPhoto 1"
             "Photo",
             "Photo",
             "Photo",
             "Photo 1",
             "Photo 1",
             "Photo 1"],
    (12, 2): ["Photo", # "TakingPhoto"
             "Photo 2",
             "Photo 1",
             "Photo 1",
             "Photo",
             "Photo",
             "Photo"],
    (13, 1): ["Waiting 1",
             "Waiting 1",
             "Waiting 3",
             "Waiting 1",
             "Waiting 1",
             "Waiting 1",
             "Waiting 1"],
    (13, 2): ["Waiting",
             "Waiting 2",
             "Waiting",
             "Waiting 2",
             "Waiting",
             "Waiting",
             "Waiting"],
    (14, 1): ["Walking 1",
             "Walking 1",
             "Walking 1",
             "Walking 1",
             "Walking 1",
             "Walking 1",
             "Walking 1"],
    (14, 2): ["Walking",
             "Walking",
             "Walking",
             "Walking 2",
             "Walking",
             "Walking",
             "Walking"],
    (15, 1): ["WalkDog 1", # "WalkingDog 1"
             "WalkDog 1",
             "WalkDog 1",
             "WalkDog 1",
             "WalkDog 1",
             "WalkDog 1",
             "WalkDog 1"],
    (15, 2): ["WalkDog", # "WalkingDog"
             "WalkDog",
             "WalkDog",
             "WalkDog",
             "WalkDog",
             "WalkDog",
             "WalkDog"],
    (16, 1): ["WalkTogether 1",
             "WalkTogether 1",
             "WalkTogether 1",
             "WalkTogether 1",
             "WalkTogether 1",
             "WalkTogether 1",
             "WalkTogether 1"],
    (16, 2): ["WalkTogether",
             "WalkTogether",
             "WalkTogether",
             "WalkTogether",
             "WalkTogether 2",
             "WalkTogether",
             "WalkTogether"]
}


def generate_images(subject, action, subaction, camera_id):
    video_path = "S{}/Videos/{}.{}.mp4".format(subject, subaction_names[action, subaction][subjects.index(subject)], camera_names[camera_id])
    print(video_path)
    assert os.path.isfile(video_path), video_path

    cap = cv2.VideoCapture(video_path)
    folder = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(subject, action, subaction, camera_id)
    os.makedirs(os.path.join("train", folder))

    n_frames = 0
    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        n_frames += 1
        file_name = "{}_{:06d}.jpg".format(folder, n_frames)

        if n_frames % 5 == 1:
            new_file = os.path.join("train", folder, file_name)
            cv2.imwrite(new_file, frame)

    cap.release()

    print("{} frames extracted from s-{}-a-{}-sa-{}-ca-{}".format(n_frames, subject, action, subaction, camera_id))


if __name__ == "__main__":
    for sub in [1, 5, 6, 7, 8]:
        for act in actions:
            for subact in subactions:
                for cam in camera_names:
                    generate_images(sub, act, subact, cam)
