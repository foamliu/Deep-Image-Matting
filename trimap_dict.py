def trimap_init():
    global trimap_dict
    trimap_dict = dict()
    global status
    status = {'hit': 0, 'miss': 0}


def trimap_add(alpha, trimap):
    global trimap_dict
    key = hash(str(alpha))
    trimap_dict[key] = trimap


def trimap_get(alpha):
    key = hash(str(alpha))
    if key in trimap_dict.keys():
        status['hit'] = status['hit'] + 1
        return trimap_dict[key]
    else:
        status['miss'] = status['miss'] + 1
        return None


def trimap_clear(epoch):
    global status
    size = len(trimap_dict)
    with open("training.txt", "a") as file:
        file.write("Epoch %d, cleaning %d trimaps, hit=%d, miss=%d.\n" % (epoch, size, status['hit'], status['miss']))
    trimap_dict.clear()
    status = {'hit': 0, 'miss': 0}


if __name__ == '__main__':
    pass
