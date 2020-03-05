from neuralcorefres.parsedata.preco_parser import *


def main():
    data = [get_preco_data(PreCoDataType.TEST)[0]]
    data = prep_for_nn(data)

    print("\n\n")
    for key, value in data.items():
        print("KEY:", key)
        [print(f"\t{c.__str__()}") for c in value]


if __name__ == "__main__":
    main()
