#include <neural_core/model_zoo/model_card.h>
#include <stdio.h>

/*
 * SNEPPX - Simple Test
 *
 * WHAT
 *   Simple Test.
 *
 * CONCEPT
 *   Provides the Simple Test.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


int main(void) {
    printf("Starting test...\n");
    
    ModelCard *card = model_card_create();
    printf("Created card: %p\n", (void*)card);
    
    if (card) {
        model_card_set_name(card, "test-model");
        printf("Set name: %s\n", card->name);
        
        model_card_destroy(card);
        printf("Destroyed card\n");
    }
    
    printf("Test complete\n");
    return 0;
}
