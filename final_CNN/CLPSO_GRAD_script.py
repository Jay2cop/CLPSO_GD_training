#parallelization
def worker(particle, model_path, device, val_loader, criterion, fitnesses, index):
    fitness = evaluate(particle, model_path, device, val_loader, criterion)
    fitnesses[index] = fitness
    print(f"Particle {index + 1} evaluated. Fitness: {fitness:0.4f}")

#main CLPSO with gradient
def run_clpso(model_path, val_loader, criterion, fine_tune_epochs=2, num_particles=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, c1, c2 = 0.8125244572938359, 1.5504439576831914, 1.21405147410338
    bounds = 0.1
    gd_learning_rate = 0.09828401086555535
    gd_weight_decay = 0.004320915587110087

    # Load the model to determine its structure
    model = my_CNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc2.parameters():
        param.requires_grad = True

    model.fc2.reset_parameters()  # Randomize weights of fully connected layer
    model = model.to(device)
    num_output_neurons = 10  # Assuming 10 output classes
    num_ftrs = model.fc2.in_features
    total_params = (num_ftrs * num_output_neurons) + num_output_neurons

    # Initialize particles and their velocities
    particles = [np.random.uniform(-1, 1, total_params) for _ in range(num_particles)]
    velocities = [np.zeros(total_params) for _ in range(num_particles)]
    personal_best_positions = [np.copy(p) for p in particles]
    personal_best_scores = [float('inf') for _ in range(num_particles)]
    global_best_position = np.random.uniform(-1, 1, total_params)
    global_best_score = float('inf')

    # Early stopping
    early_stopping = EarlyStopping(patience=5)
    optimizer = torch.optim.SGD(model.fc2.parameters(), lr=gd_learning_rate, weight_decay=gd_weight_decay)
    manager = mp.Manager()
    fitnesses = manager.list([0] * num_particles)

    for epoch in range(fine_tune_epochs):
        print(f"Epoch {epoch+1}/{fine_tune_epochs}")
        for i in range(num_particles):
            r1, r2 = np.random.rand(total_params), np.random.rand(total_params)
            for d in range(total_params):
                selected_particle = np.random.choice(num_particles)  # Select a random particle for each dimension
                velocities[i][d] = w * velocities[i][d] + c1 * r1[d] * (personal_best_positions[selected_particle][d] - particles[i][d]) + c2 * r2[d] * (global_best_position[d] - particles[i][d])
                velocities[i][d] = np.clip(velocities[i][d], -bounds, bounds)  # Apply velocity clipping

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)  # Apply bounds to particle position
             
            # Load updated particle position into model
            particle_position_tensor = torch.from_numpy(particles[i]).float().to(device)
            weight_part = particle_position_tensor[:-10].view_as(model.fc2.weight)
            bias_part = particle_position_tensor[-10:]
            model.fc2.weight.data.copy_(weight_part)
            model.fc2.bias.data.copy_(bias_part)

            # Gradient Descent Update for particle i
            optimizer.zero_grad()
            model.train()
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()

            # Extract updated position from model parameters
            with torch.no_grad():
                particles[i] = np.concatenate([model.fc2.weight.data.view(-1).cpu().numpy(), model.fc2.bias.data.cpu().numpy()])
        
        processes = []
        for i, particle in enumerate(particles):
            p = mp.Process(target=worker, args=(particle, model_path, device, val_loader, criterion, fitnesses, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Update personal and global bests
        for i, fitness in enumerate(fitnesses):
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particles[i].copy()
                print(f"New global best fitness: {global_best_score:.4f}")
        with torch.no_grad():
            global_best_tensor = torch.from_numpy(global_best_position).float().to(device)
            weight_part = global_best_tensor[:-10].view_as(model.fc2.weight)
            bias_part = global_best_tensor[-10:]
            model.fc2.weight.data.copy_(weight_part)
            model.fc2.bias.data.copy_(bias_part)

        # Validation and Early Stopping
        model.eval()
        val_loss = 0.0
        all_predict = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predict = torch.max(outputs, 1)
                all_predict.extend(predict.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        precision = precision_score(all_labels, all_predict, average='weighted', zero_division=1)

        print(f"Epoch {epoch + 1}/{fine_tune_epochs} - Validation Loss: {val_loss:0.4f}, Precision: {precision:0.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        print(f"Epoch {epoch+1}/{fine_tune_epochs} - Best Global Fitness: {global_best_score:0.4f}")

    print(f"Optimization completed. Best Global Fitness: {global_best_score:0.4f}")
    return model