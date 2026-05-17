package com.example.smartparkingbackend.controller;

import com.example.smartparkingbackend.model.OccupancyHistory;
import com.example.smartparkingbackend.service.OccupancyService;
import org.springframework.web.bind.annotation.*;

import java.util.List;


@RestController
@RequestMapping("/api/parking")
@CrossOrigin
public class OccupancyController {

    private final OccupancyService occupancyService;

    public OccupancyController(OccupancyService occupancyService) {
        this.occupancyService = occupancyService;
    }

    @GetMapping("/{id}/current")
    public OccupancyHistory getCurrentOccupancy(@PathVariable String id) {
        return occupancyService.getLatestOccupancy(id);
    }

    @GetMapping("/{id}/history")
    public List<OccupancyHistory> getOccupancyHistory(@PathVariable String id) {
        return occupancyService.getHistory(id);
    }
}
