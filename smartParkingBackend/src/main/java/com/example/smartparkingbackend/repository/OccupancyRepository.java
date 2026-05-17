package com.example.smartparkingbackend.repository;

import com.example.smartparkingbackend.model.OccupancyHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface OccupancyRepository extends JpaRepository<OccupancyHistory, Long> {
    List<OccupancyHistory> findTop1ByParkingIdOrderByTimestampDesc(String parkingId);

    List<OccupancyHistory> findByParkingIdOrderByTimestampDesc(String parkingId);
}
