def Merger(self):
        
        pairs = ct.detect_collisions(positions)
        
        for i,j in pairs:
            if masses[i]==0 or masses[j]==0:
                continue
            big,small = (self.sprites_by_id[i],self.sprites_by_id[j]) if masses[i] > masses[j] else (self.sprites_by_id[j],self.sprites_by_id[i])
            mass_b=masses[big.index]
            mass_s=masses[small.index]
            
            velocities[big.index] = (mass_b*velocities[big.index]+mass_s*velocities[small.index,None])/(mass_b+mass_s)
            positions[big.index] = (mass_b*positions[big.index]+mass_s*positions[small.index,None])/(mass_b+mass_s)
            masses[big.index] += mass_s
            
            velocities[small.index] = (0,0)
            positions[small.index] =(0,0)
            masses[small.index] = 0
            
            big.update_texture(int(((big.radius**3)+(small.radius**3))**(1/3)),big.color)
            
            self.deleted.append(small.index)
            self.planets.remove(small)