#include <Kokkos_Core.hpp>
#include <iostream>
#include "serial_moc.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

    Kokkos::parallel_for("hello", 10, KOKKOS_LAMBDA(int i) {
      printf("Hello from i = %d\n", i);
    });
  }

  double keff = serial_moc_sweep(argc, argv);

  Kokkos::finalize();
  return 0;
}
