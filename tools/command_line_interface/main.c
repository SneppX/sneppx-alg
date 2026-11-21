/*
 * SNEPPX - Main
 *
 * WHAT
 *   Main.
 *
 * CONCEPT
 *   Provides the Main.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * SNEPPX Command-Line Interface — SKELETON
 * VERSION: v0.5
 *
 * PURPOSE: Main CLI entry point for training, evaluation, export, and
 * system inspection.  Uses a subcommand dispatch table.
 *
 * Subcommands:
 *   train   — train a model from a config file
 *   eval    — evaluate a checkpoint on a dataset
 *   export  — export model to ONNX / native format
 *   benchmark — run performance benchmarks
 *   inspect — print system info (devices, memory, version)
 *
 * Usage: SNEPPX <command> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    int (*handler)(int argc, char** argv);
    const char* description;
} SNEPPXCommand;

static int cmd_train(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("train: not yet implemented\n");
    return 0;
}

static int cmd_eval(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("eval: not yet implemented\n");
    return 0;
}

static int cmd_export(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("export: not yet implemented\n");
    return 0;
}

static int cmd_benchmark(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("benchmark: not yet implemented\n");
    return 0;
}

static int cmd_inspect(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("inspect: not yet implemented\n");
    return 0;
}

static const SNEPPXCommand commands[] = {
    {"train", cmd_train, "Train a model from a config file"},
    {"eval", cmd_eval, "Evaluate a checkpoint on a dataset"},
    {"export", cmd_export, "Export model to ONNX / native format"},
    {"benchmark", cmd_benchmark, "Run performance benchmarks"},
    {"inspect", cmd_inspect, "Print system information"},
    {NULL, NULL, NULL},
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: SNEPPX <command> [options]\n\nCommands:\n");
        for (int i = 0; commands[i].name; i++)
            fprintf(stderr, "  %-12s %s\n", commands[i].name, commands[i].description);
        return 1;
    }
    for (int i = 0; commands[i].name; i++) {
        if (strcmp(argv[1], commands[i].name) == 0)
            return commands[i].handler(argc - 1, argv + 1);
    }
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
