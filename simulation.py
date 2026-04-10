import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def simulate_market(num_traders=1000, prob_conn=0.01, num_steps=50000, initial_margin=100, shock_volatility=20, contagion_volatility=15):
    # Network of traders (Erdos-Renyi) representing correlated strategies or herd behavior
    G = nx.erdos_renyi_graph(num_traders, prob_conn)
    margins = np.ones(num_traders) * initial_margin
    
    avalanches = []
    
    for _ in range(num_steps):
        # Noise: Random trading order/news impact (shock to a random trader)
        shocked_trader = np.random.randint(0, num_traders)
        shock_size = np.absolute(np.random.normal(0, shock_volatility))
        margins[shocked_trader] -= shock_size
        
        # Avalanche: Check for margin calls
        failed_this_step = 0
        to_check = [shocked_trader]
        failed_set = set()
        
        while to_check:
            trader = to_check.pop(0)
            if margins[trader] < 0 and trader not in failed_set:
                failed_this_step += 1
                failed_set.add(trader)
                margins[trader] = initial_margin  # Reset after liquidation (to keep system alive)
                
                # Connectivity: Herd behavior/correlated assets, shock neighbors
                for neighbor in G.neighbors(trader):
                    if neighbor not in failed_set:
                        margins[neighbor] -= np.absolute(np.random.normal(0, contagion_volatility))
                        if margins[neighbor] < 0:
                            to_check.append(neighbor)
        
        if failed_this_step > 0:
            avalanches.append(failed_this_step)
            
    return avalanches

print("Simulating Low Connectivity Market...")
avalanches_low = simulate_market(prob_conn=0.003, num_steps=50000)
print("Simulating High Connectivity Market...")
avalanches_high = simulate_market(prob_conn=0.008, num_steps=50000)

# Plotting frequency distribution
def plot_pdf(avalanches, label, marker, color):
    counts, bins = np.histogram(avalanches, bins=np.logspace(0, np.log10(max(avalanches)+1), 20))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    probs = counts / len(avalanches)
    # Filter out zeros for log-log plot
    idx = probs > 0
    plt.plot(bin_centers[idx], probs[idx], marker, linestyle='', label=label, color=color, alpha=0.8)

plt.figure(figsize=(8, 6))

if avalanches_low:
    plot_pdf(avalanches_low, label='Low Connectivity ($p=0.003$)', marker='o', color='blue')
if avalanches_high:
    plot_pdf(avalanches_high, label='High Connectivity ($p=0.008$)', marker='s', color='red')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Avalanche Size $s$ (Number of Margin Calls)', fontsize=12)
plt.ylabel('Distribution $P(s) \propto s^{-\\tau}$', fontsize=12)
plt.title('Self-Organized Criticality in Financial Markets', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('avalanche_distribution.png', dpi=300)
print("Saved figure to avalanche_distribution.png")
