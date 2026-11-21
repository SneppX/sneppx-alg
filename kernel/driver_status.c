#include "neural_core/drivers/driver_status.h"
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



const char* sneppx_driver_status_string(sneppx_driver_status_t status) {
    switch (status) {
        case SNEPPX_DRIVER_OK:
            return "ok";
        case SNEPPX_DRIVER_UNSUPPORTED:
            return "backend not supported on this build/runtime";
        case SNEPPX_DRIVER_ERROR:
        default:
            return "backend error";
    }
}
