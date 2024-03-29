Algorithm: CLPSO with Gradient Descent for CNN Optimization
Input: model_path, val_loader, criterion, fine_tune_epochs, num_particles

Define device, w, c1, c2, bounds, gd_learning_rate, gd_weight_decay, p_threshold
Load CNN model from model_path onto device
Freeze all layers except the last layer of the CNN

Initialize particles, velocities, personal_best_positions, personal_best_scores
Set global_best_position and global_best_score

For each epoch from 1 to fine_tune_epochs:
    For each particle i from 0 to num_particles - 1:
        For each dimension d from 0 to total_params - 1:
            Generate random numbers r1, r2
            If random number < p_threshold:
                Select learning_source from another particle's best position
            Else:
                Select learning_source from particle i's best position
            Calculate and clip velocity for dimension d
        Update particle position and clip within bounds
        Load updated particle position as CNN weights for last layer
        Perform Gradient Descent update on CNN
        Calculate and store fitness for particle i
        Update personal and global bests based on fitness
    Check for early stopping condition
    Calculate and display epoch loss and precision

Return optimized CNN model, along with epoch losses and precisions