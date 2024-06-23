import os

train_dir_labels = "/Users/riccardoderitis/Magistrale/1° Anno/CV & DL/progetto/Aerial Person Detection/train/labels"
train_dir_imgs = "/Users/riccardoderitis/Magistrale/1° Anno/CV & DL/progetto/Aerial Person Detection/train/images"
val_dir_labels = "/Users/riccardoderitis/Magistrale/1° Anno/CV & DL/progetto/New Aerial/valid/labels"
val_dir_imgs = "/Users/riccardoderitis/Magistrale/1° Anno/CV & DL/progetto/New Aerial/valid/images"

file_removed: str = []
os.chdir(val_dir_labels)
os.remove(f"{val_dir_labels}/.DS_Store")
list_file = os.listdir()
for f in list_file:
    txt = open(f, "r+")
    lines = []
    for line in txt.read().split("\n"):
        if int(line.split(" ")[0]) == 4:
            lines.append(line)
    txt.close()
    if lines != []:
        txt_w = open(f, "w")
        for line in lines:
            labels = line.split(" ")
            labels[0] = 0
            if lines[-1] == line:
                txt_w.write(" ".join(str(x) for x in labels))
            else:
                txt_w.write(" ".join(str(x) for x in labels)+"\n")
        txt_w.close()
    else:
        file_removed.append(f)

for file in file_removed:
    os.remove(f"{val_dir_labels}/{file}")
    file_img = file.replace(".txt", ".jpg")
    os.remove(f"{val_dir_imgs}/{file_img}")
