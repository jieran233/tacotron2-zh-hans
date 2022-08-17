import sys

from text import cleaners

if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit()
    print(cleaners.zh_hans_cleaners(sys.argv[1]))