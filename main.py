from processing import processing
from ray_tune import optimize
from final_run import final_run

def main():
    
    processing()
    optimize()
    final_run()
    
    print("Success!")

if __name__ == "__main__":
    main()
