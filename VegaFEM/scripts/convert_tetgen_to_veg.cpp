#include <cstdio>
#include <cstdlib>
#include <vegafem/tetMesh.h>

using namespace vegafem;

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::fprintf(stderr, "Usage: %s <tetgen basename> <out .veg>\n", argv[0]);
    std::fprintf(stderr, "Example: %s liver_HD_Low liver_HD_Low.veg\n", argv[0]);
    return 1;
  }

  const char *basename = argv[1];
  const char *outVeg = argv[2];

  // specialFileType=0 => load TetGen/Stellar ".node" + ".ele" from basename
  TetMesh tetMesh(basename, /*specialFileType=*/0, /*verbose=*/1);
  int code = tetMesh.saveToAscii(outVeg);
  if (code != 0)
  {
    std::fprintf(stderr, "Error: failed to write %s\n", outVeg);
    return 2;
  }

  std::printf("Wrote %s\n", outVeg);
  return 0;
}

