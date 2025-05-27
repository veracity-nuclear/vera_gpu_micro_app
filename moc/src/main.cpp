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

  const std::vector<std::string> args(argv, argv + argc);
  double keff = serial_moc_sweep(args);

  Kokkos::finalize();
  return 0;
}
