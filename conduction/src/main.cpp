#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

    Kokkos::parallel_for("hello", 10, KOKKOS_LAMBDA(int i) {
      printf("Hello from i = %d\n", i);
    });
  }
  Kokkos::finalize();
  return 0;
}
