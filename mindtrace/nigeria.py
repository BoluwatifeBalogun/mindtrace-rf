"""Nigeria's 36 states and the FCT, grouped by geopolitical zone."""
ZONES = {
    "North Central": ["Benue", "FCT Abuja", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}
STATES = sorted(s for states in ZONES.values() for s in states)
ZONE_OF = {s: z for z, states in ZONES.items() for s in states}
ROLES = ["Radiographer", "Radiologist", "Neurologist", "Medical officer", "Researcher", "Student"]
