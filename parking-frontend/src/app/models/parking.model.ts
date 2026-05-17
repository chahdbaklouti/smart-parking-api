// Interfaces pour les données du parking

export interface Camera {
  id: string;
  name: string;
  video: string;
  hasAnnotation: boolean;
}

export interface CameraListResponse {
  cameras: Camera[];
  total: number;
}

export interface ParkingStats {
  parking_id: string;
  occupied: number;
  available: number;
  total: number;
  occupancy_rate: number;
  timestamp: string;
  frame?: number;
}

export interface GlobalStats {
  parkings: ParkingStats[];
  global: {
    total_occupied: number;
    total_available: number;
    total_spots: number;
    occupancy_rate: number;
  };
  timestamp: string;
}

// Pour Spring Boot
export interface ParkingInfo {
  id: string;
  name: string;
  capacity: number;
  currentOccupied: number;
  currentAvailable: number;
  occupancyRate: number;
}

export interface SpringGlobalStats {
  parkings: ParkingInfo[];
  totalOccupied: number;
  totalAvailable: number;
  totalCapacity: number;
  globalOccupancyRate: number;
  timestamp: string;
}

export interface TimeSlotStats {
  slot: number;
  label: string;
  avgOccupancyRate: number;
  avgOccupied: number;
  avgAvailable: number;
}
