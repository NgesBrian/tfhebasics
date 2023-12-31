#include <tfhe.h>

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

int main(void)
{
    int ok = 0;
    // Prepare the config builder for the high level API and choose which types to enable
    ConfigBuilder *builder;
    Config *config;

    // Put the builder in a default state without any types enabled
    config_builder_all_disabled(&builder);
    // Enable the uint128 type using the small LWE key for encryption
    config_builder_enable_default_uint128_small(&builder);
    // Populate the config
    config_builder_build(builder, &config);

    ClientKey *client_key = NULL;
    ServerKey *server_key = NULL;

    // Generate the keys using the config
    generate_keys(config, &client_key, &server_key);
    // Set the server key for the current thread
    set_server_key(server_key);

    FheUint128 *lhs = NULL;
    FheUint128 *rhs = NULL;
    FheUint128 *result = NULL;

    // Encrypt a u128 using 64 bits words, we encrypt 20 << 64 | 10
    ok = fhe_uint128_try_encrypt_with_client_key_u128(10, 20, client_key, &lhs);
    assert(ok == 0);

    // Encrypt a u128 using words, we encrypt 2 << 64 | 1
    ok = fhe_uint128_try_encrypt_with_client_key_u128(1, 2, client_key, &rhs);
    assert(ok == 0);

    // Compute the subtraction
    ok = fhe_uint128_sub(lhs, rhs, &result);
    assert(ok == 0);

    uint64_t w0, w1;
    // Decrypt
    ok = fhe_uint128_decrypt(result, client_key, &w0, &w1);
    assert(ok == 0);

    // Here the subtraction allows us to compare each word
    assert(w0 == 9);
    assert(w1 == 18);

    // Destroy the ciphertexts
    fhe_uint128_destroy(lhs);
    fhe_uint128_destroy(rhs);
    fhe_uint128_destroy(result);

    // Destroy the keys
    client_key_destroy(client_key);
    server_key_destroy(server_key);
    return EXIT_SUCCESS;
}
