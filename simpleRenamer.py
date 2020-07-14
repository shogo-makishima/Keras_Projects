import os, xml.etree.ElementTree as ElementTree, cv2

def RenameFiles(dir: str, type_file="jpg", zeroID = 0, maxNumbers = 4):
    list_files = list(filter(lambda x: x if x[-len(type_file):] == type_file else None, os.listdir(dir)))
    for i in range(len(list_files)):
        os.rename(f"{dir}\\{list_files[i]}", f"{dir}\\{str('0' * (4 - len(str(zeroID + i + 1))) + f'{zeroID + i + 1}')}.{type_file}")

def ChangeAnnotationFiles(dir: str, type_file="xml"):
    xml_files = list(filter(lambda x: x if x[-len(type_file):] == type_file else None, os.listdir(dir)))
    for i in range(len(xml_files)):
        xml = ElementTree.parse(f"{dir}\\{xml_files[i]}")

        root = xml.getroot()

        root[1].text = f"{xml_files[i]}"
        root[2].text = f"{dir}\\{xml_files[i]}"

        xml.write(f"{dir}\\{xml_files[i]}")

def ResizeImage(dir: str, size=(128, 128), type_file="jpg"):
    list_files = list(filter(lambda x: x if x[-len(type_file):] == type_file else None, os.listdir(dir)))
    for i in range(len(list_files)):
        image = cv2.imread(f"{dir}\\{list_files[i]}")
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(f"{dir}\\{list_files[i]}", gray)


# ResizeImage("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized\\train")

# Rename train
# RenameFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima\\train", "jpg")
# RenameFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima\\train", "xml")

# Rename test
RenameFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized\\test", "jpg", 22)
RenameFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized\\test", "xml", 22)

# Change train xml files
# ChangeAnnotationFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima\\train", "xml")

# Change test xml files
ChangeAnnotationFiles("I:\GitHub\Shogo-Makishima\Datasets\Shogo-Makishima-Resized\\test", "xml")
