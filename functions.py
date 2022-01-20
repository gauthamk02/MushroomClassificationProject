['class', 'gill-color', 'spore-print-color', 'population', 'gill-size',
        'stalk-root', 'habitat']

agaricus_bisporus = {'gill-color': 'b', 'spore-print-color': 'n', 'gill-size': 'n',
'habitat': 'g', 'population': 'a', 'stalk-root': 'c'}


def classify():
    bruises = input("Bruise [bruises=t,no=f]: ")
    gill_size = input("Gill Size [broad=b,narrow=n]: ")
    gill_color = input("Gill Color [black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y]: ")
    stalk_shape = input("Stalk Shape [enlarging=e,tapering=t]: ")
    stalk_root = input("Stalk Root [bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?]: ")
    spore_print_color = input("Spore print color [black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y]: ")
    population = input("Population [abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y]: ")
    habitat = input("Habitat [grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d]: ")

