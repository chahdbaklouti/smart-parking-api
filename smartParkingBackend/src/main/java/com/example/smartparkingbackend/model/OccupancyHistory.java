package com.example.smartparkingbackend.model;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Table (name = "occupancy_history")
@Data
public class OccupancyHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "parking_id")
    private String parkingId;

    private LocalDateTime timestamp;

    @Column(name = "free_spots")
    private int availableSpots;

    @Column(name = "capacity")
    private int capacity;

    @Column(name = "occupancy_rate")
    private float occupancyRate;

    @Column(name = "hour")
    private int hour;

    @Column(name = "is_weekend")
    private boolean isWeekend;

    @Column(name = "free_spot_ids")
    private String free_spots_list;

    @Column(name = "is_special_day")
    private boolean is_special_day;

}
