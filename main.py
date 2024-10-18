import cv2
import neat
import csv
import numpy as np
import pickle  # Import pickle for saving the model

# Setup CSV file
with open('neat_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Generation', 'Genome_ID', 'Fitness', 'Motor1_Speed', 'Motor2_Speed', 'Servo_Speed'])

def get_motor_speeds():
    motor1_speed = 0.5
    motor2_speed = 0.6
    return motor1_speed, motor2_speed

def get_servo_speed():
    servo_speed = 0.7
    return servo_speed

def get_camera_image():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
    except:
        # Mock image (e.g., black image of expected size)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Adjust size as needed
    return frame

def get_map_data():
    map_data = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    return map_data

def eval_genomes(genomes, config, generation):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        motor1_speed, motor2_speed = get_motor_speeds()
        servo_speed = get_servo_speed()
        try:
            camera_image = get_camera_image()
        except RuntimeError as e:
            print(e)
            genome.fitness = 0.0
            continue
        map_data = get_map_data()
        
        gray_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 64))
        flattened_image = resized_image.flatten().tolist()
        
        inputs = [motor1_speed, motor2_speed, servo_speed] + [item for sublist in map_data for item in sublist] + flattened_image
        
        outputs = net.activate(inputs)
        
        # Placeholder values for fitness calculation - make sure these change
        distance_travelled = np.random.rand() * 100  # Simulating variability
        time_elapsed = np.random.rand() * 10
        smoothness_penalty = np.random.rand()
        proximity_to_target = np.random.rand() * 50
        crash_car_collision = np.random.rand() * 1000
        genome.fitness = distance_travelled + time_elapsed - smoothness_penalty - proximity_to_target

        # Store data in CSV
        with open('neat_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, genome_id, genome.fitness, motor1_speed, motor2_speed, servo_speed])

def run_evolution(config_path, generations):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    for generation in range(generations):
        population.run(lambda genomes, config: eval_genomes(genomes, config, generation))
        
        # Get the best genome and mutate it
        best_genome = max(population.population.values(), key=lambda g: g.fitness)
        new_genome = pickle.loads(pickle.dumps(best_genome))  # Deep copy the best genome
        new_genome.mutate(config.genome_config)  # Mutate the copied genome
        population.population[new_genome.key] = new_genome  # Add the mutated genome back into the population

    return population.best_genome

config_path = "config-feedforward.ini"
generations = 3
winner = run_evolution(config_path, generations)
print(f'Best genome:\n{winner}')

# Save the best genome
with open('best_genome.pkl', 'wb') as f:
    pickle.dump(winner, f)
