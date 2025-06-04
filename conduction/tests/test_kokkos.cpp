#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {
void run_kokkos_test() {

    // Print a message to verify Kokkos is working
    Kokkos::printf("Hello from Kokkos!\n");

    const int N = 5;
    Kokkos::View<int*> results("results", N);

    Kokkos::parallel_for("FillArray", N, KOKKOS_LAMBDA(const int i) {
        results(i) = i * i;
    });

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(host_results(i), i * i);
    }
}

TEST(KokkosTest, BasicTest) {
    run_kokkos_test();
}
