def trimap_init():
    global trimap_dict
    trimap_dict = dict()
    global status
    status = dict()
    status['hit'] = 0
    status['miss'] = 0
    status['add'] = 0
    status['addDup'] = 0


def trimap_add(alpha, trimap):
    key = hash(str(alpha))
    if key in trimap_dict.keys():
        status['addDup'] += 1

    trimap_dict[key] = trimap
    status['add'] += 1


def trimap_get(alpha):
    key = hash(str(alpha))
    if key in trimap_dict.keys():
        status['hit'] += 1
        return trimap_dict[key]
    else:
        status['miss'] += 1
        return None


def trimap_clear(epoch):
    size = len(trimap_dict)
    with open("training.txt", "a") as file:
        file.write("Epoch %d, cleaning %d trimaps, hit=%d, miss=%d, add=%d, addDup=%d\n" % (epoch, size, status['hit'], status['miss'], status['add'], status['addDup']))
    trimap_dict.clear()
    status['hit'] = 0
    status['miss'] = 0
    status['add'] = 0
    status['addDup'] = 0


if __name__ == '__main__':
    trimap_init()

    trimap_add([1, 2, 3], [3, 2, 1])
    trimap_add([1, 2, 3], [4, 5, 6])
    trimap = trimap_get([1, 2, 3])
    print(trimap)
    trimap = trimap_get([1, 2, 2])
    trimap_clear(1)
